'use client';

import { useCallback, useRef, useState, type ChangeEvent, type DragEvent, type ReactNode } from 'react';
import { AlertTriangle, ArrowRight, Loader2, RotateCcw, UploadCloud } from 'lucide-react';
import { useVideoAnalysis, type UseVideoAnalysisResult } from '@/hooks/useVideoAnalysis';
import { MetricCard, PHASE_BADGE_CLASSES, METRIC_CAPTIONS } from './VideoAnalyzer';
import { PHASE_LABELS, SPRINT_PHASES, type SessionSummary, type SprintPhase } from '@/utils/biomechanics';

// ---------------------------------------------------------------------------
// One independent upload + video + live-metrics column
// ---------------------------------------------------------------------------

interface VideoSlotProps {
  title: string;
  analysis: UseVideoAnalysisResult;
}

function VideoSlot({ title, analysis }: VideoSlotProps): ReactNode {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const {
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
    handleVideoPlay,
    handleVideoPause,
    handleVideoError,
    handleVideoSeeked,
    loadVideo,
    onFileInputChange,
    reset,
  } = analysis;

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsDragging(false);
      const file = event.dataTransfer.files?.[0];
      if (file) loadVideo(file);
    },
    [loadVideo]
  );

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
  }, []);

  const isModelLoading = modelState === 'loading';

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">{title}</h2>
        {videoSrc && (
          <button
            type="button"
            onClick={reset}
            className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1 text-xs font-medium text-slate-300 transition-colors hover:bg-white/10"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            New Video
          </button>
        )}
      </div>

      {errorMessage && (
        <div className="flex items-start gap-2.5 rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-xs text-amber-300">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {!videoSrc ? (
        <div
          aria-label={`${title} sprint video upload area`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`flex aspect-video w-full flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed transition-colors ${
            isDragging ? 'border-sky-400 bg-sky-400/5' : 'border-white/15 bg-white/[0.02]'
          }`}
        >
          <div className="flex h-11 w-11 items-center justify-center rounded-full bg-sky-500/10 ring-1 ring-sky-400/30">
            <UploadCloud className="h-5 w-5 text-sky-400" />
          </div>
          <p className="px-4 text-center text-xs text-slate-500">Drag &amp; drop, or</p>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="rounded-lg bg-sky-500 px-3 py-1.5 text-xs font-semibold text-white outline-none transition-colors hover:bg-sky-400 focus-visible:ring-2 focus-visible:ring-sky-300"
          >
            Browse Files
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/webm,video/quicktime,.mov"
            className="hidden"
            aria-label={`Choose a sprint video file for ${title}`}
            onChange={onFileInputChange}
          />
        </div>
      ) : (
        <>
          <div className="relative w-full overflow-hidden rounded-xl border border-white/10 bg-black shadow-xl">
            <div className="relative aspect-video w-full">
              <video
                ref={videoRef}
                src={videoSrc}
                controls
                playsInline
                onPlay={handleVideoPlay}
                onPause={handleVideoPause}
                onEnded={handleVideoPause}
                onError={handleVideoError}
                onSeeked={handleVideoSeeked}
                className="absolute inset-0 h-full w-full object-contain"
              >
                <track kind="captions" />
              </video>
              <canvas
                ref={canvasRef}
                aria-hidden="true"
                className="pointer-events-none absolute inset-0 h-full w-full object-contain"
              />

              {isModelLoading && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/80 backdrop-blur-sm">
                  <Loader2 className="h-6 w-6 animate-spin text-sky-400" />
                  <p className="text-xs font-medium text-slate-200">Warming up AI model…</p>
                </div>
              )}

              {!isModelLoading && liveMetrics && (
                <div className="pointer-events-none absolute left-2 top-2">
                  <span
                    className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold backdrop-blur-sm ${PHASE_BADGE_CLASSES[liveMetrics.phase]}`}
                  >
                    {PHASE_LABELS[liveMetrics.phase]}
                  </span>
                </div>
              )}

              {!isModelLoading && isAnalyzing && isTrackingLost && (
                <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center p-2">
                  <div className="flex items-center gap-1.5 rounded-full bg-black/70 px-2.5 py-0.5 text-[10px] font-medium text-amber-300 backdrop-blur-sm">
                    <AlertTriangle className="h-3 w-3" />
                    Tracking lost
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span className="max-w-[10rem] truncate">{fileName}</span>
            <span className="text-slate-700">•</span>
            <span>{frameCount} frames</span>
            <span className="text-slate-700">•</span>
            <span className="inline-flex items-center gap-1">
              <span className={`h-1.5 w-1.5 rounded-full ${isAnalyzing ? 'animate-pulse bg-green-400' : 'bg-slate-600'}`} />
              {isAnalyzing ? 'Analyzing' : 'Paused'}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              label="Trunk Lean"
              caption={METRIC_CAPTIONS.trunkLean}
              angle={liveMetrics?.trunkLeanAngle ?? null}
              status={liveMetrics?.trunkLeanStatus ?? null}
            />
            <MetricCard
              label="Knee Drive"
              caption={METRIC_CAPTIONS.kneeDrive}
              sublabel={liveMetrics?.kneeDriveSide ? `${liveMetrics.kneeDriveSide} leg` : undefined}
              angle={liveMetrics?.kneeDriveAngle ?? null}
              status={liveMetrics?.kneeDriveStatus ?? null}
            />
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Comparison summary table — data-level comparison (phase-normalized stats),
// not a frame-synced video comparison. See note in useVideoAnalysis: two
// independent videos have no natural shared wall-clock timeline, so this
// compares each side's own per-phase summaries instead of trying to lock
// playback together.
// ---------------------------------------------------------------------------

interface MetricRow {
  label: string;
  format: (summary: SessionSummary) => string | null;
}

const METRIC_ROWS: MetricRow[] = [
  { label: 'Form Score', format: (s) => (s.overallScore !== null ? `${Math.round(s.overallScore)} / 100` : null) },
  {
    label: 'Ground Contact Time',
    format: (s) => (s.groundContactTime ? `${(s.groundContactTime.averageAngle * 1000).toFixed(0)} ms` : null),
  },
  {
    label: 'Over-stride',
    format: (s) => (s.overstride ? `${s.overstride.averageAngle.toFixed(2)} m` : null),
  },
  {
    label: 'Knee Extension at Toe-Off',
    format: (s) => (s.kneeExtensionAtToeOff ? `${s.kneeExtensionAtToeOff.averageAngle.toFixed(0)}°` : null),
  },
  {
    label: 'Cadence',
    format: (s) => (s.stepFrequency ? `${(s.stepFrequency.averageHz * 60).toFixed(0)} spm` : null),
  },
];

function ComparisonSummary({
  left,
  right,
}: {
  left: UseVideoAnalysisResult;
  right: UseVideoAnalysisResult;
}): ReactNode {
  const phasesWithData = SPRINT_PHASES.filter((phase) => left.phaseSummaries[phase] || right.phaseSummaries[phase]);

  return (
    <div>
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Comparison Summary</h2>
      {phasesWithData.length === 0 ? (
        <div className="rounded-xl border border-white/10 bg-white/5 p-5 text-sm text-slate-500">
          Play both videos to start building the comparison summary — each phase appears here once either side has
          enough analyzed frames.
        </div>
      ) : (
      <div className="space-y-4">
        {phasesWithData.map((phase) => {
          const leftSummary = left.phaseSummaries[phase];
          const rightSummary = right.phaseSummaries[phase];
          return (
            <div key={phase} className="overflow-hidden rounded-xl border border-white/10 bg-white/5">
              <div className="border-b border-white/10 px-4 py-2.5">
                <span
                  className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ${PHASE_BADGE_CLASSES[phase]}`}
                >
                  {PHASE_LABELS[phase]}
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wide text-slate-500">
                      <th className="px-4 py-2 font-medium">Metric</th>
                      <th className="px-4 py-2 font-medium">A</th>
                      <th className="px-4 py-2 font-medium">B</th>
                    </tr>
                  </thead>
                  <tbody>
                    {METRIC_ROWS.map((row) => {
                      const leftValue = leftSummary ? row.format(leftSummary) : null;
                      const rightValue = rightSummary ? row.format(rightSummary) : null;
                      if (leftValue === null && rightValue === null) return null;
                      return (
                        <tr key={row.label} className="border-t border-white/5">
                          <td className="px-4 py-2 text-slate-400">{row.label}</td>
                          <td className="px-4 py-2 font-medium text-slate-200 tabular-nums">{leftValue ?? '—'}</td>
                          <td className="px-4 py-2 font-medium text-slate-200 tabular-nums">{rightValue ?? '—'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })}
      </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function ComparisonView() {
  const left = useVideoAnalysis();
  const right = useVideoAnalysis();

  return (
    <div className="w-full space-y-8">
      <div className="flex items-start gap-3 rounded-xl border border-sky-400/20 bg-sky-400/5 p-4 text-sm text-sky-100">
        <ArrowRight className="mt-0.5 h-4 w-4 shrink-0 text-sky-400" />
        <p>
          Load two clips — a before/after, or left vs. right leg — and each is analyzed independently. Since two
          different videos have no shared wall-clock timeline, the two aren&apos;t frame-synced; instead the summary
          below compares each side&apos;s own per-phase form scores, contact times, and over-stride once both have
          enough analyzed frames.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <VideoSlot title="Video A" analysis={left} />
        <VideoSlot title="Video B" analysis={right} />
      </div>

      <ComparisonSummary left={left} right={right} />
    </div>
  );
}
