'use client';

import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';

/**
 * Records the annotated sprint video (raw frames + the skeleton/joint-arc
 * overlay) to a downloadable WebM file.
 *
 * MediaRecorder can only capture a canvas that's being drawn to live — it
 * can't retroactively re-encode frames that already played. So "export" is
 * implemented as a fresh replay of the source video from t=0 with recording
 * turned on: `startExport()` resets the analysis state, seeks to the start,
 * and calls `video.play()`, which fires the video's existing `onPlay` handler
 * exactly as a normal user-initiated playback would — restarting the same
 * MediaPipe inference / canvas-drawing pipeline unchanged. This hook's only
 * new work is compositing the video frame + the (transparent) overlay canvas
 * onto a third, never-mounted canvas each animation frame, and feeding that
 * composite's captured stream into a MediaRecorder.
 */

const MIME_TYPE_CANDIDATES = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm'];

function pickSupportedMimeType(): string {
  if (typeof MediaRecorder === 'undefined') return 'video/webm';
  return MIME_TYPE_CANDIDATES.find((type) => MediaRecorder.isTypeSupported(type)) ?? 'video/webm';
}

export interface UseVideoExportOptions {
  videoRef: RefObject<HTMLVideoElement | null>;
  overlayCanvasRef: RefObject<HTMLCanvasElement | null>;
  /** Clears the analyzer's accumulated per-frame state so the export replay starts from a clean slate, same as loading a fresh video. */
  onResetAnalysisState: () => void;
  fps?: number;
}

export interface UseVideoExportResult {
  isSupported: boolean;
  isExporting: boolean;
  /** 0..1, driven by video.currentTime / video.duration during the replay. */
  exportProgress: number;
  exportedUrl: string | null;
  errorMessage: string | null;
  startExport: () => void;
  cancelExport: () => void;
}

export function useVideoExport({
  videoRef,
  overlayCanvasRef,
  onResetAnalysisState,
  fps = 30,
}: UseVideoExportOptions): UseVideoExportResult {
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportedUrl, setExportedUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const compositeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const rafRef = useRef<number | null>(null);
  const exportedUrlRef = useRef<string | null>(null);
  const cleanupListenersRef = useRef<(() => void) | null>(null);

  const isSupported =
    typeof window !== 'undefined' &&
    typeof MediaRecorder !== 'undefined' &&
    typeof HTMLCanvasElement.prototype.captureStream === 'function';

  const stopCompositeLoop = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const compositeTick = useCallback(() => {
    const video = videoRef.current;
    const overlay = overlayCanvasRef.current;
    const composite = compositeCanvasRef.current;

    if (!video || !overlay || !composite || video.paused || video.ended) {
      rafRef.current = null;
      return;
    }

    const ctx = composite.getContext('2d');
    if (ctx) {
      if (composite.width !== video.videoWidth || composite.height !== video.videoHeight) {
        composite.width = video.videoWidth;
        composite.height = video.videoHeight;
      }
      ctx.drawImage(video, 0, 0, composite.width, composite.height);
      ctx.drawImage(overlay, 0, 0, composite.width, composite.height);
    }

    if (video.duration > 0) {
      setExportProgress(Math.min(1, video.currentTime / video.duration));
    }

    rafRef.current = requestAnimationFrame(compositeTick);
  }, [videoRef, overlayCanvasRef]);

  const finishExport = useCallback(() => {
    stopCompositeLoop();
    cleanupListenersRef.current?.();
    cleanupListenersRef.current = null;

    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();
    } else {
      setIsExporting(false);
    }
  }, [stopCompositeLoop]);

  const cancelExport = useCallback(() => {
    videoRef.current?.pause();
    finishExport();
  }, [videoRef, finishExport]);

  const startExport = useCallback(() => {
    const video = videoRef.current;
    if (!video || !isSupported || isExporting) return;

    if (exportedUrlRef.current) {
      URL.revokeObjectURL(exportedUrlRef.current);
      exportedUrlRef.current = null;
    }
    setExportedUrl(null);
    setErrorMessage(null);

    if (!compositeCanvasRef.current) {
      compositeCanvasRef.current = document.createElement('canvas');
    }
    const composite = compositeCanvasRef.current;
    composite.width = video.videoWidth || 1280;
    composite.height = video.videoHeight || 720;

    onResetAnalysisState();
    chunksRef.current = [];
    setExportProgress(0);
    setIsExporting(true);

    const canvasStream = composite.captureStream(fps);

    let audioTracks: MediaStreamTrack[] = [];
    try {
      const videoWithCapture = video as HTMLVideoElement & { captureStream?: () => MediaStream };
      audioTracks = videoWithCapture.captureStream?.().getAudioTracks() ?? [];
    } catch {
      audioTracks = [];
    }

    const stream = new MediaStream([...canvasStream.getVideoTracks(), ...audioTracks]);
    const mimeType = pickSupportedMimeType();

    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(stream, { mimeType });
    } catch (err) {
      console.error('StrideSight: failed to start video export recorder', err);
      setErrorMessage('Could not start recording in this browser.');
      setIsExporting(false);
      return;
    }
    mediaRecorderRef.current = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mimeType });
      const url = URL.createObjectURL(blob);
      exportedUrlRef.current = url;
      setExportedUrl(url);
      setExportProgress(1);
      setIsExporting(false);
    };

    // A 'pause' event covers both the user pausing mid-export and playback
    // reaching the end (the HTML spec fires 'pause' immediately before
    // 'ended' in that case) — either way, stop recording rather than
    // continuing to capture a frozen frame.
    const handlePause = () => finishExport();
    video.addEventListener('pause', handlePause);
    cleanupListenersRef.current = () => video.removeEventListener('pause', handlePause);

    video.currentTime = 0;
    recorder.start();
    void video.play();
    rafRef.current = requestAnimationFrame(compositeTick);
  }, [videoRef, isSupported, isExporting, onResetAnalysisState, fps, compositeTick, finishExport]);

  useEffect(() => {
    return () => {
      stopCompositeLoop();
      cleanupListenersRef.current?.();
      if (exportedUrlRef.current) URL.revokeObjectURL(exportedUrlRef.current);
    };
  }, [stopCompositeLoop]);

  return { isSupported, isExporting, exportProgress, exportedUrl, errorMessage, startExport, cancelExport };
}
