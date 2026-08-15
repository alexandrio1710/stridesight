'use client';

import { useCallback, useEffect, useRef, useState, type ChangeEvent } from 'react';
import type { Pose as PoseInstance, Results } from '@mediapipe/pose';
import { drawAngleArcs, drawSkeleton, type Connection } from '@/utils/canvasDrawing';
import { type GaitEventRecord } from '@/utils/chartData';
import {
  addSampleInPlace,
  calculateOverstrideDistance,
  calculateTrunkLeanAngle,
  classifyByLowerBound,
  classifyByTargetRange,
  classifyByUpperBound,
  classifyPhaseWithHysteresis,
  computeFrameMetrics,
  computePhaseSummaries,
  createPhaseAggregates,
  downloadCSV,
  generateCSV,
  isGroundContactPeak,
  isLandmarkReliable,
  isStepPeak,
  isToeOffPeak,
  recordFrameMetrics,
  smoothValue,
  smoothValueOverTime,
  CADENCE_MAX_HZ,
  CADENCE_MIN_HZ,
  FLIGHT_TIME_MAX_SECONDS,
  FLIGHT_TIME_MIN_SECONDS,
  GROUND_CONTACT_TIME_MAX_SECONDS,
  GROUND_CONTACT_TIME_MIN_SECONDS,
  PHASE_THRESHOLDS,
  POSE_LANDMARK_INDICES,
  TOE_OFF_SEARCH_WINDOW_SECONDS,
  TRUNK_LEAN_PHASE_TIME_CONSTANT_SECONDS,
  type AngleSample,
  type FrameMetrics,
  type PhaseAggregates,
  type PoseLandmarks,
  type SessionSummary,
  type Side,
  type SprintPhase,
} from '@/utils/biomechanics';
import { useVideoExport, type UseVideoExportResult } from './useVideoExport';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export type ModelState = 'idle' | 'loading' | 'ready' | 'error';
export type PhaseSelection = 'auto' | SprintPhase;

const MEDIAPIPE_CDN_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/pose';
export const ACCEPTED_MIME_TYPES = ['video/mp4', 'video/webm', 'video/quicktime'];
export const ACCEPTED_EXTENSION_PATTERN = /\.(mp4|webm|mov)$/i;

/** How aggressively the live-display numbers are smoothed. Raw values (unsmoothed) still drive
 * aggregation, canvas drawing, and step detection — this only softens what the eye sees. */
const DISPLAY_SMOOTHING_ALPHA = 0.35;

interface DisplayEma {
  trunkLean: number | null;
  kneeDrive: number | null;
  hipExtension: number | null;
  leftArm: number | null;
  rightArm: number | null;
}

function createDisplayEma(): DisplayEma {
  return { trunkLean: null, kneeDrive: null, hipExtension: null, leftArm: null, rightArm: null };
}

/**
 * Produces a display-only copy of a frame's metrics with the five angles
 * lightly smoothed and re-classified against the same phase thresholds.
 * MediaPipe's own `smoothLandmarks` option already removes most raw jitter,
 * but a small extra pass here keeps the numbers in the dashboard from
 * flickering between adjacent status colors on borderline frames — a purely
 * cosmetic concern that has no business affecting the actual recorded data.
 */
function smoothDisplayMetrics(ema: DisplayEma, raw: FrameMetrics): FrameMetrics {
  const thresholds = PHASE_THRESHOLDS[raw.phase];

  const next = (prev: number | null, value: number | null): number | null =>
    value === null ? null : prev === null ? value : smoothValue(prev, value, DISPLAY_SMOOTHING_ALPHA);

  ema.trunkLean = next(ema.trunkLean, raw.trunkLeanAngle);
  ema.kneeDrive = next(ema.kneeDrive, raw.kneeDriveAngle);
  ema.hipExtension = next(ema.hipExtension, raw.hipExtensionAngle);
  ema.leftArm = next(ema.leftArm, raw.leftArmSwingAngle);
  ema.rightArm = next(ema.rightArm, raw.rightArmSwingAngle);

  return {
    ...raw,
    trunkLeanAngle: ema.trunkLean,
    trunkLeanStatus:
      ema.trunkLean !== null
        ? classifyByTargetRange(
            ema.trunkLean,
            thresholds.trunkLean.optimalMin,
            thresholds.trunkLean.optimalMax,
            thresholds.trunkLean.cautionMin,
            thresholds.trunkLean.cautionMax
          )
        : null,
    kneeDriveAngle: ema.kneeDrive,
    kneeDriveStatus:
      ema.kneeDrive !== null
        ? classifyByUpperBound(ema.kneeDrive, thresholds.kneeDrive.optimalMax, thresholds.kneeDrive.cautionMax)
        : null,
    hipExtensionAngle: ema.hipExtension,
    hipExtensionStatus:
      ema.hipExtension !== null
        ? classifyByLowerBound(
            ema.hipExtension,
            thresholds.hipExtension.optimalMin,
            thresholds.hipExtension.cautionMin
          )
        : null,
    leftArmSwingAngle: ema.leftArm,
    leftArmSwingStatus:
      ema.leftArm !== null
        ? classifyByTargetRange(
            ema.leftArm,
            thresholds.armSwing.optimalMin,
            thresholds.armSwing.optimalMax,
            thresholds.armSwing.cautionMin,
            thresholds.armSwing.cautionMax
          )
        : null,
    rightArmSwingAngle: ema.rightArm,
    rightArmSwingStatus:
      ema.rightArm !== null
        ? classifyByTargetRange(
            ema.rightArm,
            thresholds.armSwing.optimalMin,
            thresholds.armSwing.optimalMax,
            thresholds.armSwing.cautionMin,
            thresholds.armSwing.cautionMax
          )
        : null,
  };
}

// ---------------------------------------------------------------------------
// Inference-frame downscaling: shrinks the frame handed to MediaPipe so the
// (CPU-bound, WASM) pose model has far fewer pixels to process per call --
// a direct win for frame rate on typical 1080p+ phone footage, independent
// of anything else. Only the offscreen copy handed to MediaPipe is
// downscaled; the visible <video>/<canvas> overlay still render at full
// source resolution, and MediaPipe's landmark coordinates are normalized to
// [0,1] regardless of input pixel dimensions, so nothing downstream needs
// to change.
//
// This was originally added while chasing a `RuntimeError: Aborted(native
// code called abort())` crash from the legacy @mediapipe/pose WASM binary,
// reproduced against real 1920x1080 phone footage. Isolating the cause
// showed the crash tracks the software (SwiftShader) WebGL renderer used by
// headless test automation, not frame resolution: drawing that same 1080p
// video down to this canvas's resolution still crashed every frame, while a
// natively re-encoded small file (same pixel dimensions, decoded start to
// finish at that size) ran error-free. That points at the video element's
// own full-resolution decode/GL path, not the size of what's ultimately
// handed to pose.send(), which this function does not touch. So this
// downscale is kept for its real performance benefit, not as a confirmed
// crash fix -- verify against a real GPU-accelerated browser before relying
// on it to prevent that crash for actual users.
//
// Web Worker note: offloading MediaPipe inference to a worker was evaluated
// (see project history) and ruled out — @mediapipe/pose 0.5.x's internal
// asset loader doesn't correctly honor `locateFile` for all its assets
// inside a worker's global scope (one binary asset resolves relative to the
// worker script's own URL instead, and the library calls `importScripts()`
// on a non-JS `.tflite` file, which browsers reject outright). Inference
// stays on the main thread; this downscale is the main lever for keeping it
// responsive.
// ---------------------------------------------------------------------------

const MAX_INFERENCE_DIMENSION = 640;

function getDownscaledInferenceFrame(
  canvasRef: { current: HTMLCanvasElement | null },
  video: HTMLVideoElement
): HTMLCanvasElement | null {
  const { videoWidth, videoHeight } = video;
  if (videoWidth === 0 || videoHeight === 0) return null;

  const longerSide = Math.max(videoWidth, videoHeight);
  if (longerSide <= MAX_INFERENCE_DIMENSION) return null;

  const scale = MAX_INFERENCE_DIMENSION / longerSide;
  const targetWidth = Math.round(videoWidth * scale);
  const targetHeight = Math.round(videoHeight * scale);

  if (!canvasRef.current) {
    canvasRef.current = document.createElement('canvas');
  }
  const canvas = canvasRef.current;
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
  return canvas;
}

// ---------------------------------------------------------------------------
// Gait event detection: ground contact and toe-off, tracked per anatomical
// leg (unlike kneeDriveAngle/kneeDriveSide, which follow whichever leg
// currently leads). Each side needs its own continuous rolling buffers
// since ground contact and toe-off happen on independent schedules per leg.
// ---------------------------------------------------------------------------

interface GaitEventState {
  ankleYBuffer: AngleSample[];
  kneeAngleBuffer: AngleSample[];
  lastGroundContactTimestamp: number | null;
  lastToeOffTimestamp: number | null;
  /** Timestamp of the most recent unpaired ground contact, while still within the toe-off search window. */
  awaitingToeOffSince: number | null;
  /**
   * Sprint phase active at that same ground contact — a stance phase can, in
   * principle, straddle a phase transition (contact under one phase,
   * toe-off confirmed a few frames later under the next). Ground contact
   * time and knee extension at toe-off both describe that single stance
   * phase, so both get attributed to the phase it *started* in, not
   * whichever phase happens to be active on the later frame toe-off is
   * confirmed on.
   */
  phaseAtGroundContact: SprintPhase | null;
}

function createGaitEventState(): GaitEventState {
  return {
    ankleYBuffer: [],
    kneeAngleBuffer: [],
    lastGroundContactTimestamp: null,
    lastToeOffTimestamp: null,
    awaitingToeOffSince: null,
    phaseAtGroundContact: null,
  };
}

/**
 * Updates one leg's gait-event state with this frame's ankle height (2D
 * image y) and knee angle (3D world), mutating `state` and feeding
 * `phaseAggregates[currentPhase].overstride` /
 * `phaseAggregates[currentPhase].flightTime` (attributed to the phase the
 * new contact lands in) and `phaseAggregates[<phase at contact>]
 * .kneeExtensionAtToeOff` / `.groundContactTime` (attributed to the phase
 * the stance *started* in — see the GaitEventState field doc) in place when
 * a ground-contact or toe-off event is confirmed.
 *
 * The over-stride distance is sampled from *this* frame's world landmarks
 * at the moment ground contact is confirmed — which is one frame after the
 * actual peak, since the local-max detector needs to see the following
 * sample to confirm a peak. At typical video frame rates that's well under
 * 17ms of lag, negligible for a slow-changing hip-to-ankle distance, so this
 * is a deliberate, documented approximation rather than buffering a history
 * of past world-landmark frames just to look one frame back.
 */
function processGaitEvents(
  state: GaitEventState,
  side: Side,
  ankleY: number | null,
  kneeAngle: number | null,
  timestampSeconds: number,
  worldLandmarks: PoseLandmarks,
  currentPhase: SprintPhase,
  phaseAggregates: PhaseAggregates,
  eventHistory: GaitEventRecord[]
): void {
  const aggregates = phaseAggregates[currentPhase];

  if (ankleY !== null) {
    state.ankleYBuffer.push({ angle: ankleY, timestampSeconds });
    if (state.ankleYBuffer.length > 3) state.ankleYBuffer.shift();

    if (
      state.ankleYBuffer.length === 3 &&
      isGroundContactPeak(
        [state.ankleYBuffer[0], state.ankleYBuffer[1], state.ankleYBuffer[2]],
        state.lastGroundContactTimestamp
      )
    ) {
      const contactTimestamp = state.ankleYBuffer[1].timestampSeconds;

      // Flight time is this leg's *previous* toe-off to *this* contact — must
      // be read before lastGroundContactTimestamp/awaitingToeOffSince below
      // are overwritten for the new cycle. Bounded by FLIGHT_TIME_MAX_SECONDS
      // so a toe-off missed several strides ago can't produce a nonsensical
      // multi-stride "flight time."
      if (state.lastToeOffTimestamp !== null) {
        const flightTime = contactTimestamp - state.lastToeOffTimestamp;
        if (flightTime >= FLIGHT_TIME_MIN_SECONDS && flightTime <= FLIGHT_TIME_MAX_SECONDS) {
          addSampleInPlace(aggregates.flightTime, flightTime);
        }
      }

      state.lastGroundContactTimestamp = contactTimestamp;
      state.awaitingToeOffSince = contactTimestamp;
      state.phaseAtGroundContact = currentPhase;

      const overstride = calculateOverstrideDistance(worldLandmarks, side);
      if (overstride !== null) {
        addSampleInPlace(aggregates.overstride, overstride);
      }
      eventHistory.push({
        type: 'groundContact',
        side,
        timestampSeconds: contactTimestamp,
        value: overstride,
        phase: currentPhase,
      });
    }
  }

  if (kneeAngle !== null) {
    state.kneeAngleBuffer.push({ angle: kneeAngle, timestampSeconds });
    if (state.kneeAngleBuffer.length > 3) state.kneeAngleBuffer.shift();

    const stillWaiting =
      state.awaitingToeOffSince !== null &&
      timestampSeconds - state.awaitingToeOffSince <= TOE_OFF_SEARCH_WINDOW_SECONDS;

    if (
      stillWaiting &&
      state.kneeAngleBuffer.length === 3 &&
      isToeOffPeak(
        [state.kneeAngleBuffer[0], state.kneeAngleBuffer[1], state.kneeAngleBuffer[2]],
        state.lastToeOffTimestamp
      )
    ) {
      const toeOffSample = state.kneeAngleBuffer[1];
      // Both attributed to phaseAtGroundContact, not currentPhase — see the
      // GaitEventState field doc for why a stance phase that straddles a
      // phase transition belongs to the phase it started in.
      const strideStartPhase = state.phaseAtGroundContact ?? currentPhase;
      const strideAggregates = phaseAggregates[strideStartPhase];
      // state.awaitingToeOffSince is this same cycle's ground-contact
      // timestamp (set above, untouched since) — the stance duration.
      const groundContactTime = toeOffSample.timestampSeconds - state.awaitingToeOffSince!;
      if (groundContactTime >= GROUND_CONTACT_TIME_MIN_SECONDS && groundContactTime <= GROUND_CONTACT_TIME_MAX_SECONDS) {
        addSampleInPlace(strideAggregates.groundContactTime, groundContactTime);
      }
      state.lastToeOffTimestamp = toeOffSample.timestampSeconds;
      state.awaitingToeOffSince = null;
      state.phaseAtGroundContact = null;
      addSampleInPlace(strideAggregates.kneeExtensionAtToeOff, toeOffSample.angle);
      eventHistory.push({
        type: 'toeOff',
        side,
        timestampSeconds: toeOffSample.timestampSeconds,
        value: toeOffSample.angle,
        phase: strideStartPhase,
      });
    } else if (
      state.awaitingToeOffSince !== null &&
      timestampSeconds - state.awaitingToeOffSince > TOE_OFF_SEARCH_WINDOW_SECONDS
    ) {
      // Window expired without a confirmed toe-off — stop waiting so a
      // stale "awaiting" flag can't block pairing the *next* ground contact.
      state.awaitingToeOffSince = null;
      state.phaseAtGroundContact = null;
    }
  }
}

// ---------------------------------------------------------------------------
// Public hook
// ---------------------------------------------------------------------------

export interface UseVideoAnalysisResult {
  videoRef: React.MutableRefObject<HTMLVideoElement | null>;
  canvasRef: React.MutableRefObject<HTMLCanvasElement | null>;

  modelState: ModelState;
  errorMessage: string | null;
  videoSrc: string | null;
  fileName: string;
  liveMetrics: FrameMetrics | null;
  frameCount: number;
  isTrackingLost: boolean;
  isAnalyzing: boolean;
  hasAttemptedPlayback: boolean;
  phaseSummaries: Partial<Record<SprintPhase, SessionSummary>>;
  phaseOverride: PhaseSelection;
  /** Bumped (not read directly) a few times a second during analysis — re-read frameHistoryRef/gaitEventHistoryRef on change. */
  chartRevision: number;
  /** Full per-frame history so far. A ref (not state) to avoid a re-render per frame — snapshot with `[...frameHistoryRef.current]` when chartRevision changes. */
  frameHistoryRef: React.MutableRefObject<FrameMetrics[]>;
  gaitEventHistoryRef: React.MutableRefObject<GaitEventRecord[]>;

  handleVideoPlay: () => void;
  handleVideoPause: () => void;
  handleVideoError: () => void;
  handleVideoSeeked: () => void;
  handlePhaseOverrideChange: (value: PhaseSelection) => void;

  /** Validates the file, loads it, and resets all accumulated analysis state. */
  loadVideo: (file: File) => void;
  onFileInputChange: (event: ChangeEvent<HTMLInputElement>) => void;
  /** Clears the loaded video entirely (back to the upload prompt). */
  reset: () => void;
  handleDownloadCSV: () => void;

  videoExport: UseVideoExportResult;
}

/**
 * Owns one independent MediaPipe Pose pipeline: model lifecycle, the
 * rAF-driven per-frame inference loop, gait-event/phase tracking, and the
 * accumulated frame history/summaries. Multiple components can each call
 * this hook to run fully independent analyses side by side (see
 * ComparisonView) — everything below is local `useRef`/`useState`, so there
 * is no shared module-level state between instances. The one caveat from
 * running two instances at once: with inference on the main thread (see the
 * Web Worker note above `getDownscaledInferenceFrame`), two simultaneous
 * `pose.send()` loops interleave on one CPU core and roughly halve each
 * video's effective analysis speed — still correct, just slower.
 */
export function useVideoAnalysis(): UseVideoAnalysisResult {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const inferenceCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const poseRef = useRef<PoseInstance | null>(null);
  const poseConnectionsRef = useRef<readonly Connection[]>([]);
  const cancelledRef = useRef(false);
  const loopRunningRef = useRef(false);
  const frameHistoryRef = useRef<FrameMetrics[]>([]);
  const frameIndexRef = useRef(0);
  const objectUrlRef = useRef<string | null>(null);
  const phaseAggregatesRef = useRef<PhaseAggregates>(createPhaseAggregates());
  const trunkLeanEmaRef = useRef<number | null>(null);
  // Real elapsed time since the last trunk-lean EMA update, not frame count
  // — see smoothValueOverTime() for why. Reset alongside trunkLeanEmaRef.
  const trunkLeanEmaTimestampRef = useRef<number | null>(null);
  const previousPhaseRef = useRef<SprintPhase | null>(null);
  const kneeAngleBufferRef = useRef<AngleSample[]>([]);
  const lastStepPeakTimestampRef = useRef<number | null>(null);
  const gaitEventStateRef = useRef<Record<Side, GaitEventState>>({
    left: createGaitEventState(),
    right: createGaitEventState(),
  });
  const gaitEventHistoryRef = useRef<GaitEventRecord[]>([]);
  // Wall-clock time (performance.now()) of the last chart re-render, so the
  // dashboard updates a few times a second during analysis instead of on
  // every frame — frameHistoryRef/gaitEventHistoryRef stay refs (no re-render
  // cost of their own); this timestamp only gates how often the throttled
  // `chartRevision` tick fires to read a fresh snapshot of them.
  const lastChartUpdateRef = useRef(0);
  const displayEmaRef = useRef<DisplayEma>(createDisplayEma());
  // Read inside handleResults (a stable-identity callback) rather than a
  // dependency, so toggling the phase override doesn't force handleResults
  // to be recreated — which would re-run the MediaPipe init effect and
  // reload the entire WASM model just to flip a UI toggle.
  const phaseOverrideRef = useRef<PhaseSelection>('auto');
  // The video's currentTime at the moment a frame was actually handed to
  // pose.send(), captured in processFrame(). handleResults() runs
  // asynchronously once MediaPipe finishes that frame's inference — reading
  // video.currentTime *there* instead would give whatever position playback
  // has since advanced to, not the position the inferred frame was actually
  // captured at. Under any real inference latency (worse under slow/CPU-only
  // WASM execution, but nonzero even in the best case) that skews every
  // timestamp-derived measurement: trunk-lean smoothing's dt, step
  // frequency, and especially ground contact time / flight time, whose
  // entire measured quantity is a sub-200ms interval where even one frame of
  // drift is a significant fraction of the value being measured.
  const pendingFrameTimestampRef = useRef(0);

  const [modelState, setModelState] = useState<ModelState>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [liveMetrics, setLiveMetrics] = useState<FrameMetrics | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [isTrackingLost, setIsTrackingLost] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  // Distinguishes "hasn't pressed play yet" from "played the whole clip and
  // got zero usable frames" -- both look identical as frameCount === 0 &&
  // !isAnalyzing, but they call for very different messaging.
  const [hasAttemptedPlayback, setHasAttemptedPlayback] = useState(false);
  const [phaseSummaries, setPhaseSummaries] = useState<Partial<Record<SprintPhase, SessionSummary>>>({});
  const [phaseOverride, setPhaseOverride] = useState<PhaseSelection>('auto');
  const [chartRevision, setChartRevision] = useState(0);

  // -- MediaPipe results callback -------------------------------------------

  const handleResults = useCallback((results: Results) => {
    if (cancelledRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || video.videoWidth === 0) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const landmarks = results.poseLandmarks as PoseLandmarks | undefined;
    // Real-world 3D coordinates (meters, origin at hip midpoint), same
    // 33-point indexing as `landmarks`. Falls back to an empty array (rather
    // than undefined) so downstream per-joint reliability checks degrade
    // safely to "no data" instead of throwing if a frame is somehow missing it.
    const worldLandmarks = (results.poseWorldLandmarks as PoseLandmarks | undefined) ?? [];

    const frameTimestamp = pendingFrameTimestampRef.current;

    if (landmarks && landmarks.length > 0) {
      const rawLean = calculateTrunkLeanAngle(landmarks);
      if (rawLean !== null) {
        const previousTimestamp = trunkLeanEmaTimestampRef.current;
        const dt = previousTimestamp !== null ? Math.max(0, frameTimestamp - previousTimestamp) : 0;
        trunkLeanEmaRef.current = smoothValueOverTime(
          trunkLeanEmaRef.current,
          rawLean,
          dt,
          TRUNK_LEAN_PHASE_TIME_CONSTANT_SECONDS
        );
        trunkLeanEmaTimestampRef.current = frameTimestamp;
      }

      // Default to 'transition' (the neutral middle phase) on the rare frame
      // where lean can't be computed yet at all — e.g. the very first frame
      // before the torso is confidently tracked.
      const autoPhase: SprintPhase =
        trunkLeanEmaRef.current !== null
          ? classifyPhaseWithHysteresis(trunkLeanEmaRef.current, previousPhaseRef.current)
          : 'transition';
      const phase = phaseOverrideRef.current === 'auto' ? autoPhase : phaseOverrideRef.current;
      // Tracks the auto-detector's own history for hysteresis, independent
      // of `phase` above — a manual phase override must not corrupt the
      // auto-classifier's state, or switching back to "Auto-detect" would
      // resume hysteresis from whatever phase the user had manually picked.
      previousPhaseRef.current = autoPhase;

      const metrics = computeFrameMetrics(
        landmarks,
        worldLandmarks,
        frameIndexRef.current,
        frameTimestamp,
        phase
      );
      frameIndexRef.current += 1;
      frameHistoryRef.current.push(metrics);
      recordFrameMetrics(phaseAggregatesRef.current, metrics);

      // Step-cadence detection: the knee-drive angle oscillates once per
      // step (each leg's drive peak alternates with the other's), so a
      // rolling 3-sample local-maximum check on that raw signal gives a
      // real-time, causal step frequency without any extra tracking.
      if (metrics.kneeDriveAngle !== null) {
        const buffer = kneeAngleBufferRef.current;
        buffer.push({ angle: metrics.kneeDriveAngle, timestampSeconds: metrics.timestampSeconds });
        if (buffer.length > 3) buffer.shift();

        if (buffer.length === 3 && isStepPeak([buffer[0], buffer[1], buffer[2]], lastStepPeakTimestampRef.current)) {
          const peakTimestamp = buffer[1].timestampSeconds;
          if (lastStepPeakTimestampRef.current !== null) {
            const interval = peakTimestamp - lastStepPeakTimestampRef.current;
            const stepFrequencyHz = interval > 0 ? 1 / interval : 0;
            // Sanity clamp: reject anything outside a plausible *sprinting* cadence.
            if (stepFrequencyHz > CADENCE_MIN_HZ && stepFrequencyHz < CADENCE_MAX_HZ) {
              addSampleInPlace(phaseAggregatesRef.current[phase].stepFrequency, stepFrequencyHz);
            }
          }
          lastStepPeakTimestampRef.current = peakTimestamp;
        }
      }

      // Ground-contact / toe-off detection, tracked independently per
      // anatomical leg (see the GaitEventState doc for why this can't reuse
      // kneeDriveAngle, which follows whichever leg currently leads).
      const leftAnkle = landmarks[POSE_LANDMARK_INDICES.LEFT_ANKLE];
      const rightAnkle = landmarks[POSE_LANDMARK_INDICES.RIGHT_ANKLE];

      processGaitEvents(
        gaitEventStateRef.current.left,
        'left',
        isLandmarkReliable(leftAnkle) ? leftAnkle.y : null,
        metrics.leftKneeAngle,
        metrics.timestampSeconds,
        worldLandmarks,
        phase,
        phaseAggregatesRef.current,
        gaitEventHistoryRef.current
      );
      processGaitEvents(
        gaitEventStateRef.current.right,
        'right',
        isLandmarkReliable(rightAnkle) ? rightAnkle.y : null,
        metrics.rightKneeAngle,
        metrics.timestampSeconds,
        worldLandmarks,
        phase,
        phaseAggregatesRef.current,
        gaitEventHistoryRef.current
      );

      setLiveMetrics(smoothDisplayMetrics(displayEmaRef.current, metrics));
      setFrameCount(frameHistoryRef.current.length);
      setPhaseSummaries(computePhaseSummaries(phaseAggregatesRef.current));

      // Throttled to a few times a second (not every frame) — the dashboard
      // charts don't need frame-perfect live updates, and re-rendering
      // Recharts at 30-60fps would be wasted work during analysis.
      const now = performance.now();
      if (now - lastChartUpdateRef.current > 300) {
        lastChartUpdateRef.current = now;
        setChartRevision((r) => r + 1);
      }

      drawSkeleton(ctx, landmarks, poseConnectionsRef.current, canvas.width, canvas.height);
      drawAngleArcs(ctx, landmarks, metrics, canvas.width, canvas.height);
      setIsTrackingLost(false);
    } else {
      setLiveMetrics(null);
      setIsTrackingLost(true);
    }
  }, []);

  // -- Frame processing loop (recursive async rAF, video-file driven) ------

  const processFrame = useCallback(async () => {
    const video = videoRef.current;
    const pose = poseRef.current;

    if (!video || !pose || cancelledRef.current || video.paused || video.ended) {
      loopRunningRef.current = false;
      return;
    }

    try {
      const inferenceFrame = getDownscaledInferenceFrame(inferenceCanvasRef, video);
      pendingFrameTimestampRef.current = video.currentTime;
      await pose.send({ image: inferenceFrame ?? video });
    } catch (err) {
      console.error('StrideSight: pose estimation error on frame', err);
    }

    const stillPlaying = videoRef.current && !videoRef.current.paused && !videoRef.current.ended;
    if (!cancelledRef.current && stillPlaying) {
      requestAnimationFrame(() => {
        void processFrame();
      });
    } else {
      loopRunningRef.current = false;
    }
  }, []);

  const handleVideoPlay = useCallback(() => {
    if (!loopRunningRef.current && poseRef.current && modelState === 'ready') {
      loopRunningRef.current = true;
      setIsAnalyzing(true);
      setHasAttemptedPlayback(true);
      void processFrame();
    }
  }, [modelState, processFrame]);

  const handleVideoPause = useCallback(() => {
    loopRunningRef.current = false;
    setIsAnalyzing(false);
    // Force one final, un-throttled chart update so the last frames analyzed
    // before pausing show up immediately rather than waiting for the next
    // 300ms tick that will now never come (the throttle only fires from
    // inside handleResults, which stops running once the loop is paused).
    setChartRevision((r) => r + 1);
  }, []);

  const handleVideoError = useCallback(() => {
    setErrorMessage(
      "This video couldn't be played. Your browser may not support its codec — this is common with HEVC-encoded .mov files from iPhones. Try re-exporting as H.264 MP4, or open this page in Safari."
    );
  }, []);

  // Scrubbing the timeline while paused doesn't run pose estimation (only
  // 'play' drives the analysis loop), so without this the skeleton overlay
  // would keep showing a stale pose from wherever the user seeked *from*,
  // overlaid on the frame they seeked *to*. Clearing it is simpler and less
  // misleading than trying to run one-off inference just for a paused seek.
  const handleVideoSeeked = useCallback(() => {
    if (!videoRef.current?.paused) return;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const handlePhaseOverrideChange = useCallback((value: PhaseSelection) => {
    phaseOverrideRef.current = value;
    setPhaseOverride(value);
  }, []);

  // -- MediaPipe lifecycle ---------------------------------------------------

  useEffect(() => {
    // `isActive` is local to *this* effect invocation, unlike a ref which is
    // shared across every invocation for the component's lifetime. That
    // distinction matters here: React 18 Strict Mode runs this effect twice
    // in development (mount -> cleanup -> mount) to surface exactly this
    // class of bug. If cancellation were tracked on a shared ref, the second
    // invocation's `cancelledRef.current = false` would silently "resurrect"
    // the first invocation's in-flight promise chain, letting it construct
    // and register a second, orphaned Pose/WASM instance that never gets
    // closed. A closure-local flag can't be resurrected by a later run.
    let isActive = true;
    let poseInstance: PoseInstance | null = null;

    cancelledRef.current = false;
    setModelState('loading');
    setErrorMessage(null);

    // Loaded dynamically (never statically imported) because @mediapipe/pose
    // touches `window`/`self` at module-evaluation time. A static top-level
    // import would execute that code during Next.js's server-side render
    // pass — even inside a 'use client' file — and crash with
    // "window is not defined". A dynamic import() only ever runs here,
    // inside a useEffect, which is guaranteed client-only.
    import('@mediapipe/pose')
      .then(async (mod) => {
        if (!isActive) return;

        poseConnectionsRef.current = mod.POSE_CONNECTIONS as unknown as Connection[];

        const pose = new mod.Pose({
          locateFile: (file) => `${MEDIAPIPE_CDN_BASE}/${file}`,
        });

        pose.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        pose.onResults(handleResults);

        if (!isActive) {
          void pose.close();
          return;
        }

        poseInstance = pose;
        poseRef.current = pose;

        await pose.initialize();

        if (!isActive) {
          void pose.close();
          poseRef.current = null;
          poseInstance = null;
          return;
        }

        setModelState('ready');
      })
      .catch((err: unknown) => {
        if (!isActive) return;
        console.error('StrideSight: failed to initialize MediaPipe Pose', err);
        setModelState('error');
        setErrorMessage('Failed to load the AI model. Check your connection and reload the page.');
      });

    return () => {
      isActive = false;
      cancelledRef.current = true;
      loopRunningRef.current = false;
      poseRef.current = null;
      void poseInstance?.close();
    };
  }, [handleResults]);

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  // -- Analysis state reset --------------------------------------------------
  // Wipes every accumulated-analysis ref/state field back to a fresh-video
  // baseline, without touching file/video-source concerns (videoSrc,
  // fileName, objectUrlRef, errorMessage). Shared by loadVideo, reset, and
  // useVideoExport's replay-from-t=0 — all three need the exact same "forget
  // everything analyzed so far" behavior, and keeping one copy means a
  // future new ref can't be added to one reset path and silently missed in
  // the others.
  const resetAnalysisState = useCallback(() => {
    frameHistoryRef.current = [];
    frameIndexRef.current = 0;
    loopRunningRef.current = false;
    phaseAggregatesRef.current = createPhaseAggregates();
    trunkLeanEmaRef.current = null;
    trunkLeanEmaTimestampRef.current = null;
    previousPhaseRef.current = null;
    kneeAngleBufferRef.current = [];
    lastStepPeakTimestampRef.current = null;
    gaitEventStateRef.current = { left: createGaitEventState(), right: createGaitEventState() };
    gaitEventHistoryRef.current = [];
    lastChartUpdateRef.current = 0;
    displayEmaRef.current = createDisplayEma();

    setLiveMetrics(null);
    setFrameCount(0);
    setIsTrackingLost(false);
    setIsAnalyzing(false);
    setHasAttemptedPlayback(false);
    setPhaseSummaries({});
    setChartRevision(0);

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  // -- File handling -----------------------------------------------------

  const loadVideo = useCallback(
    (file: File) => {
      const isAccepted =
        ACCEPTED_MIME_TYPES.includes(file.type) || ACCEPTED_EXTENSION_PATTERN.test(file.name);

      if (!isAccepted) {
        setErrorMessage('Unsupported file type. Please upload an MP4, MOV, or WebM video.');
        return;
      }

      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }

      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;

      resetAnalysisState();

      setErrorMessage(null);
      setFileName(file.name);
      setVideoSrc(url);
    },
    [resetAnalysisState]
  );

  const onFileInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) loadVideo(file);
      event.target.value = '';
    },
    [loadVideo]
  );

  const reset = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }

    resetAnalysisState();

    setVideoSrc(null);
    setFileName('');
    setErrorMessage(null);
  }, [resetAnalysisState]);

  const handleDownloadCSV = useCallback(() => {
    if (frameHistoryRef.current.length === 0) return;
    const csv = generateCSV(frameHistoryRef.current, phaseSummaries);
    const safeName = fileName.replace(/\.[^/.]+$/, '') || 'stridesight-analysis';
    downloadCSV(`${safeName}-angles.csv`, csv);
  }, [fileName, phaseSummaries]);

  // -- Annotated video export -----------------------------------------------

  const videoExport = useVideoExport({
    videoRef,
    overlayCanvasRef: canvasRef,
    onResetAnalysisState: resetAnalysisState,
  });

  return {
    videoRef,
    canvasRef,
    modelState,
    errorMessage,
    videoSrc,
    fileName,
    liveMetrics,
    frameCount,
    isTrackingLost,
    isAnalyzing,
    hasAttemptedPlayback,
    phaseSummaries,
    phaseOverride,
    chartRevision,
    frameHistoryRef,
    gaitEventHistoryRef,
    handleVideoPlay,
    handleVideoPause,
    handleVideoError,
    handleVideoSeeked,
    handlePhaseOverrideChange,
    loadVideo,
    onFileInputChange,
    reset,
    handleDownloadCSV,
    videoExport,
  };
}
