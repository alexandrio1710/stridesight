'use client';

import { useCallback, useRef, useState, type DragEvent, type ReactNode } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileVideo,
  Film,
  Info,
  Loader2,
  RotateCcw,
  Target,
  UploadCloud,
  XCircle,
} from 'lucide-react';
import BiomechanicsDashboard from './BiomechanicsDashboard';
import { statusToRGBA } from '@/utils/canvasDrawing';
import {
  ACCEPTED_EXTENSION_PATTERN,
  ACCEPTED_MIME_TYPES,
  useVideoAnalysis,
  type PhaseSelection,
} from '@/hooks/useVideoAnalysis';
import {
  generateSessionNarrative,
  MIN_SAMPLES_FOR_SUMMARY,
  MIN_STEPS_FOR_CADENCE,
  PHASE_DESCRIPTIONS,
  PHASE_LABELS,
  SPRINT_PHASES,
  type MetricStatus,
  type SessionSummary,
  type SprintPhase,
} from '@/utils/biomechanics';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STATUS_LABELS: Record<MetricStatus, string> = {
  optimal: 'Optimal',
  caution: 'Caution',
  suboptimal: 'Needs Work',
};

const OVERALL_STATUS_LABELS: Record<MetricStatus, string> = {
  optimal: 'Elite Sprint Form',
  caution: 'Solid Fundamentals — Room to Improve',
  suboptimal: 'Needs Work',
};

const PHASE_BADGE_CLASSES: Record<SprintPhase, string> = {
  acceleration: 'bg-orange-400/10 text-orange-300 ring-1 ring-inset ring-orange-400/30',
  transition: 'bg-purple-400/10 text-purple-300 ring-1 ring-inset ring-purple-400/30',
  maxVelocity: 'bg-sky-400/10 text-sky-300 ring-1 ring-inset ring-sky-400/30',
};

/** One-line "why this matters" context shown under each live metric card label. */
const METRIC_CAPTIONS = {
  trunkLean: 'Forward lean angle — the signal StrideSight uses to detect your sprint phase.',
  kneeDrive: 'How tightly the lead knee folds on the way through — front-side power. Smaller angle = tighter fold.',
  hipExtension: 'How fully the trail leg extends at push-off.',
  armSwing: 'Elbow angle through the swing — efficient arm carriage.',
} as const;

// ---------------------------------------------------------------------------
// Dashboard sub-components
// ---------------------------------------------------------------------------

interface MetricCardProps {
  label: string;
  caption: string;
  sublabel?: string;
  angle: number | null;
  status: MetricStatus | null;
}

function getStatusTextClass(status: MetricStatus | null): string {
  switch (status) {
    case 'optimal':
      return 'text-green-400';
    case 'caution':
      return 'text-yellow-400';
    case 'suboptimal':
      return 'text-red-400';
    default:
      return 'text-slate-500';
  }
}

function getStatusBadgeClass(status: MetricStatus | null): string {
  switch (status) {
    case 'optimal':
      return 'bg-green-400/10 text-green-400 ring-1 ring-inset ring-green-400/30';
    case 'caution':
      return 'bg-yellow-400/10 text-yellow-400 ring-1 ring-inset ring-yellow-400/30';
    case 'suboptimal':
      return 'bg-red-400/10 text-red-400 ring-1 ring-inset ring-red-400/30';
    default:
      return 'bg-slate-500/10 text-slate-400 ring-1 ring-inset ring-slate-500/20';
  }
}

function MetricCard({ label, caption, sublabel, angle, status }: MetricCardProps): ReactNode {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4 transition-colors">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</span>
        {sublabel && (
          <span className="rounded bg-white/5 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
            {sublabel}
          </span>
        )}
      </div>
      <p className="mt-0.5 text-[11px] leading-snug text-slate-600">{caption}</p>
      <div className={`mt-2 text-3xl font-bold tabular-nums ${getStatusTextClass(status)}`}>
        {angle !== null ? `${angle.toFixed(1)}°` : '—'}
      </div>
      <div
        className={`mt-2 inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${getStatusBadgeClass(status)}`}
      >
        {status ? STATUS_LABELS[status] : 'No data'}
      </div>
    </div>
  );
}

interface ScoreGaugeProps {
  score: number | null;
  status: MetricStatus | null;
  size?: number;
}

/**
 * Radial progress ring showing a phase's 0-100 form score. Drawn with raw
 * SVG (stroke-dasharray trick) rather than a charting dependency — it also
 * echoes the pie-slice joint arcs drawn on the video canvas, so the same
 * "arc = angle/score" visual language shows up in both places.
 */
function ScoreGauge({ score, status, size = 128 }: ScoreGaugeProps): ReactNode {
  const strokeWidth = 11;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = score !== null ? Math.min(100, Math.max(0, score)) : 0;
  const dashOffset = circumference * (1 - clamped / 100);
  const ringColor = statusToRGBA(status, 1);

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={strokeWidth}
        />
        {score !== null && (
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={ringColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            style={{ transition: 'stroke-dashoffset 0.4s ease, stroke 0.4s ease' }}
          />
        )}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-3xl font-bold tabular-nums ${getStatusTextClass(status)}`}>
          {score !== null ? Math.round(score) : '—'}
        </span>
        <span className="text-[9px] uppercase tracking-wide text-slate-500">out of 100</span>
      </div>
    </div>
  );
}

interface PhaseReportCardProps {
  phase: SprintPhase;
  summary: SessionSummary;
}

/** One self-contained score + recommendations report for a single sprint phase. */
function PhaseReportCard({ phase, summary }: PhaseReportCardProps): ReactNode {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-5">
      <span
        className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ${PHASE_BADGE_CLASSES[phase]}`}
      >
        {PHASE_LABELS[phase]}
      </span>
      <p className="mt-1.5 text-xs text-slate-500">{PHASE_DESCRIPTIONS[phase]}</p>

      <div className="mt-4 flex flex-col gap-4 sm:flex-row sm:items-center">
        <ScoreGauge score={summary.overallScore} status={summary.overallStatus} />
        <div className="flex-1">
          {summary.isReliable ? (
            <>
              <p className={`text-base font-semibold ${getStatusTextClass(summary.overallStatus)}`}>
                {summary.overallStatus ? OVERALL_STATUS_LABELS[summary.overallStatus] : ''}
              </p>
              <p className="mt-1 text-sm text-slate-500">
                Based on {summary.sampleCount} analyzed frames in this phase.
              </p>
            </>
          ) : (
            <p className="text-sm text-slate-500">
              {MIN_SAMPLES_FOR_SUMMARY - summary.sampleCount} more analyzed frames in this phase needed
              for a reliable score.
            </p>
          )}
        </div>
      </div>

      {summary.isReliable && summary.stepFrequency && summary.stepFrequency.sampleCount >= MIN_STEPS_FOR_CADENCE && (
        <div className="mt-4 border-t border-white/10 pt-4">
          <div className="flex items-baseline justify-between">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Cadence</span>
            <span className="text-sm font-semibold text-slate-200">
              {(summary.stepFrequency.averageHz * 60).toFixed(0)} spm
              {summary.stepFrequency.stdDevHz !== null && (
                <span className="text-slate-500"> ± {(summary.stepFrequency.stdDevHz * 60).toFixed(0)}</span>
              )}
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-600">
            Informational only — elite sprinters are legitimately frequency-reliant or length-reliant,
            so there is no single &ldquo;correct&rdquo; cadence to grade against.
          </p>
        </div>
      )}

      {summary.kneeExtensionAtToeOff && (
        <div className="mt-4 border-t border-white/10 pt-4">
          <div className="flex items-baseline justify-between">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Knee Extension at Toe-Off
            </span>
            <span className={`text-sm font-semibold ${getStatusTextClass(summary.kneeExtensionAtToeOff.status)}`}>
              {summary.kneeExtensionAtToeOff.averageAngle.toFixed(0)}°
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-600">
            Stance-leg hip-knee-ankle angle at push-off, from {summary.kneeExtensionAtToeOff.sampleCount}{' '}
            detected toe-offs.
          </p>
        </div>
      )}

      {summary.overstride && (
        <div className="mt-4 border-t border-white/10 pt-4">
          <div className="flex items-baseline justify-between">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Over-stride</span>
            <span className={`text-sm font-semibold ${getStatusTextClass(summary.overstride.status)}`}>
              {summary.overstride.averageAngle.toFixed(2)} m
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-600">
            Hip-to-ankle horizontal gap at footstrike, from {summary.overstride.sampleCount} detected
            ground contacts. Smaller is better — landing ahead of the hips brakes forward momentum.
          </p>
        </div>
      )}

      {summary.groundContactTime && (
        <div className="mt-4 border-t border-white/10 pt-4">
          <div className="flex items-baseline justify-between">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Ground Contact Time
            </span>
            <span className={`text-sm font-semibold ${getStatusTextClass(summary.groundContactTime.status)}`}>
              {(summary.groundContactTime.averageAngle * 1000).toFixed(0)} ms
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-600">
            Time from ground contact to toe-off, from {summary.groundContactTime.sampleCount} detected
            strides. Shorter is better — a quick, elastic contact is a hallmark of efficient sprinting.
          </p>
        </div>
      )}

      {summary.flightTime && (
        <div className="mt-4 border-t border-white/10 pt-4">
          <div className="flex items-baseline justify-between">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Flight Time</span>
            <span className="text-sm font-semibold text-slate-200">
              {(summary.flightTime.averageSeconds * 1000).toFixed(0)} ms
              {summary.flightTime.stdDevSeconds !== null && (
                <span className="text-slate-500"> ± {(summary.flightTime.stdDevSeconds * 1000).toFixed(0)}</span>
              )}
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-600">
            Time this leg spends airborne between toe-off and its next ground contact, from{' '}
            {summary.flightTime.sampleCount} detected strides. Informational only — it naturally shrinks
            alongside ground contact time as speed builds, so there is no independent target.
          </p>
        </div>
      )}

      {summary.isReliable && summary.recommendations.length > 0 && (
        <div className="mt-4 border-t border-white/10 pt-4">
          {(() => {
            const [topFocus, ...rest] = summary.recommendations;
            const isUrgent = topFocus.severity === 'suboptimal';
            return (
              <>
                <div
                  className={`flex items-start gap-2.5 rounded-lg border p-3 ${
                    isUrgent
                      ? 'border-red-400/30 bg-red-400/5'
                      : 'border-yellow-400/30 bg-yellow-400/5'
                  }`}
                >
                  <Target
                    className={`mt-0.5 h-4 w-4 shrink-0 ${isUrgent ? 'text-red-400' : 'text-yellow-400'}`}
                  />
                  <div>
                    <span
                      className={`text-[10px] font-semibold uppercase tracking-wide ${
                        isUrgent ? 'text-red-400' : 'text-yellow-400'
                      }`}
                    >
                      Top Focus
                    </span>
                    <p className="mt-0.5 text-sm text-slate-200">{topFocus.message}</p>
                  </div>
                </div>

                {rest.length > 0 && (
                  <ul className="mt-3 space-y-2.5">
                    {rest.map((rec) => (
                      <li key={rec.id} className="flex items-start gap-2.5 text-sm text-slate-300">
                        <AlertTriangle
                          className={`mt-0.5 h-4 w-4 shrink-0 ${
                            rec.severity === 'suboptimal' ? 'text-red-400' : 'text-yellow-400'
                          }`}
                        />
                        <span>{rec.message}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            );
          })()}
        </div>
      )}

      {summary.isReliable && summary.strengths.length > 0 && (
        <div className="mt-4 border-t border-white/10 pt-4">
          <span className="text-xs font-medium uppercase tracking-wide text-slate-500">What&apos;s Working</span>
          <ul className="mt-2.5 space-y-2.5">
            {summary.strengths.map((strength) => (
              <li key={strength.id} className="flex items-start gap-2.5 text-sm text-slate-300">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-green-400" />
                <span>{strength.message}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {summary.isReliable && summary.recommendations.length === 0 && summary.strengths.length === 0 && (
        <div className="mt-4 flex items-center gap-2.5 border-t border-white/10 pt-4 text-sm text-slate-500">
          <Info className="h-4 w-4 shrink-0" />
          Not enough signal yet in this phase to call out specifics.
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function VideoAnalyzer() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const analysis = useVideoAnalysis();
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

  // -- Render --------------------------------------------------------------

  const isModelLoading = modelState === 'loading';
  const hasModelError = modelState === 'error';
  const availablePhases = SPRINT_PHASES.filter((phase) => phaseSummaries[phase]);
  const sessionNarrative = generateSessionNarrative(phaseSummaries);

  return (
    <div className="w-full space-y-6">
      {hasModelError && (
        <div className="flex items-start gap-3 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-300">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {errorMessage && !hasModelError && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-500/20 bg-amber-500/10 p-4 text-sm text-amber-300">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {!videoSrc ? (
        <div
          aria-label="Sprint video upload area"
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`flex aspect-video w-full flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed transition-colors ${
            isDragging ? 'border-sky-400 bg-sky-400/5' : 'border-white/15 bg-white/[0.02]'
          }`}
        >
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-sky-500/10 ring-1 ring-sky-400/30">
            <UploadCloud className="h-7 w-7 text-sky-400" />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-slate-200">Drag &amp; drop your sprint video</p>
            <p className="mt-1 text-xs text-slate-500">MP4, MOV, or WebM — processed entirely in your browser</p>
          </div>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="rounded-lg bg-sky-500 px-4 py-2 text-sm font-semibold text-white outline-none transition-colors hover:bg-sky-400 focus-visible:ring-2 focus-visible:ring-sky-300"
          >
            Browse Files
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/webm,video/quicktime,.mov"
            className="hidden"
            aria-label="Choose a sprint video file"
            onChange={onFileInputChange}
          />
        </div>
      ) : (
        <>
          <div className="relative w-full overflow-hidden rounded-xl border border-white/10 bg-black shadow-2xl">
            <div className="relative aspect-video w-full">
              <video
                ref={videoRef}
                src={videoSrc}
                controls={!videoExport.isExporting}
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
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/80 backdrop-blur-sm">
                  <Loader2 className="h-8 w-8 animate-spin text-sky-400" />
                  <p className="text-sm font-medium text-slate-200">Warming up AI model…</p>
                </div>
              )}

              {!isModelLoading && liveMetrics && (
                <div className="pointer-events-none absolute left-3 top-3">
                  <span
                    className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold backdrop-blur-sm ${PHASE_BADGE_CLASSES[liveMetrics.phase]}`}
                  >
                    {PHASE_LABELS[liveMetrics.phase]}
                  </span>
                </div>
              )}

              {!isModelLoading && isAnalyzing && isTrackingLost && (
                <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center p-3">
                  <div className="flex items-center gap-2 rounded-full bg-black/70 px-3 py-1 text-xs font-medium text-amber-300 backdrop-blur-sm">
                    <AlertTriangle className="h-3.5 w-3.5" />
                    Tracking lost — keep the full body in frame
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-600">
            <span>Colored arcs on the video mark each tracked joint angle:</span>
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-green-400" /> optimal
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-yellow-400" /> caution
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-red-400" /> needs work
            </span>
            <span>— each judged against the currently detected phase&apos;s own targets.</span>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <FileVideo className="h-4 w-4" />
              <span className="max-w-[16rem] truncate">{fileName}</span>
              <span className="text-slate-600">•</span>
              <span>{frameCount} frames analyzed</span>
              <span className="text-slate-600">•</span>
              <span className="inline-flex items-center gap-1.5">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    isAnalyzing ? 'animate-pulse bg-green-400' : 'bg-slate-600'
                  }`}
                />
                {isAnalyzing ? 'Analyzing' : 'Paused'}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={reset}
                disabled={videoExport.isExporting}
                className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-sm font-medium text-slate-300 transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <RotateCcw className="h-4 w-4" />
                New Video
              </button>
              <button
                type="button"
                onClick={handleDownloadCSV}
                disabled={frameCount === 0 || videoExport.isExporting}
                className="inline-flex items-center gap-1.5 rounded-lg bg-sky-500 px-3 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
              >
                <Download className="h-4 w-4" />
                Download CSV
              </button>
              {videoExport.isSupported && (
                <>
                  {videoExport.isExporting ? (
                    <button
                      type="button"
                      onClick={videoExport.cancelExport}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-red-400/30 bg-red-400/10 px-3 py-1.5 text-sm font-semibold text-red-300 transition-colors hover:bg-red-400/20"
                    >
                      <XCircle className="h-4 w-4" />
                      Cancel Export ({Math.round(videoExport.exportProgress * 100)}%)
                    </button>
                  ) : videoExport.exportedUrl ? (
                    <a
                      href={videoExport.exportedUrl}
                      download={`${(fileName.replace(/\.[^/.]+$/, '') || 'stridesight-analysis')}-annotated.webm`}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-green-500 px-3 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-green-400"
                    >
                      <Download className="h-4 w-4" />
                      Save Annotated Video
                    </a>
                  ) : (
                    <button
                      type="button"
                      onClick={videoExport.startExport}
                      disabled={frameCount === 0}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-sm font-medium text-slate-300 transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Film className="h-4 w-4" />
                      Export Annotated Video
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          {videoExport.isExporting && (
            <div className="flex items-center gap-3 rounded-xl border border-sky-400/20 bg-sky-400/5 p-3 text-sm text-sky-200">
              <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
              <div className="flex-1">
                <p>Exporting annotated video&hellip; this replays the clip in real time, so it takes as long as the video&apos;s full duration.</p>
                <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full bg-sky-400 transition-[width]"
                    style={{ width: `${Math.round(videoExport.exportProgress * 100)}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {videoExport.errorMessage && (
            <div className="flex items-start gap-3 rounded-xl border border-amber-500/20 bg-amber-500/10 p-4 text-sm text-amber-300">
              <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
              <span>{videoExport.errorMessage}</span>
            </div>
          )}

          <div className="flex flex-col gap-2 rounded-xl border border-white/10 bg-white/5 p-3 sm:flex-row sm:items-center">
            <span className="shrink-0 text-xs font-medium uppercase tracking-wide text-slate-500">
              Sprint Phase
            </span>
            <div className="flex flex-wrap gap-1.5">
              {(['auto', ...SPRINT_PHASES] as PhaseSelection[]).map((option) => (
                <button
                  key={option}
                  type="button"
                  aria-pressed={phaseOverride === option}
                  onClick={() => handlePhaseOverrideChange(option)}
                  className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                    phaseOverride === option
                      ? 'bg-sky-500 text-white'
                      : 'bg-white/5 text-slate-400 hover:bg-white/10'
                  }`}
                >
                  {option === 'auto' ? 'Auto-detect' : PHASE_LABELS[option]}
                </button>
              ))}
            </div>
            <span className="text-xs text-slate-600 sm:ml-auto">
              Auto-detection reads forward trunk lean — most accurate with a side-on camera angle.
            </span>
          </div>

          {sessionNarrative && (
            <div className="rounded-xl border border-sky-400/20 bg-sky-400/5 p-4">
              <span className="text-xs font-semibold uppercase tracking-wide text-sky-400">
                Coach&apos;s Take
              </span>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-200">{sessionNarrative}</p>
            </div>
          )}

          <div>
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
              Phase Reports
            </h2>
            {availablePhases.length > 0 ? (
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3">
                {availablePhases.map((phase) => (
                  <PhaseReportCard key={phase} phase={phase} summary={phaseSummaries[phase] as SessionSummary} />
                ))}
              </div>
            ) : (
              <div className="rounded-xl border border-white/10 bg-white/5 p-5 text-sm text-slate-500">
                {hasAttemptedPlayback && !isAnalyzing && frameCount === 0 ? (
                  <>
                    <span className="text-amber-300">No pose was detected anywhere in this clip.</span>{' '}
                    Make sure a person is visible and reasonably large in frame throughout — MediaPipe
                    needs to see the full body, roughly side-on, to track joint positions. Try a closer
                    crop or a clip where the runner is clearly in view from the very first frame.
                  </>
                ) : (
                  <>
                    Play the video to start building phase-by-phase form reports. Each sprint phase
                    (acceleration, transition, max velocity) gets scored against its own biomechanical
                    targets — a clip that never reaches top speed will simply never show a max-velocity
                    card, rather than being judged against the wrong benchmark.
                  </>
                )}
              </div>
            )}
          </div>

          {frameCount > 0 && (
            <div data-chart-revision={chartRevision}>
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
                Biomechanics Charts
              </h2>
              <BiomechanicsDashboard
                frameHistory={[...frameHistoryRef.current]}
                gaitEvents={[...gaitEventHistoryRef.current]}
              />
            </div>
          )}

          <div>
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
              Live Frame Metrics
            </h2>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
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
              <MetricCard
                label="Hip Extension"
                caption={METRIC_CAPTIONS.hipExtension}
                sublabel={liveMetrics?.hipExtensionSide ? `${liveMetrics.hipExtensionSide} leg` : undefined}
                angle={liveMetrics?.hipExtensionAngle ?? null}
                status={liveMetrics?.hipExtensionStatus ?? null}
              />
              <MetricCard
                label="Left Arm Swing"
                caption={METRIC_CAPTIONS.armSwing}
                angle={liveMetrics?.leftArmSwingAngle ?? null}
                status={liveMetrics?.leftArmSwingStatus ?? null}
              />
              <MetricCard
                label="Right Arm Swing"
                caption={METRIC_CAPTIONS.armSwing}
                angle={liveMetrics?.rightArmSwingAngle ?? null}
                status={liveMetrics?.rightArmSwingStatus ?? null}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export { MetricCard, ScoreGauge, PhaseReportCard, PHASE_BADGE_CLASSES, STATUS_LABELS, OVERALL_STATUS_LABELS, METRIC_CAPTIONS, getStatusTextClass, getStatusBadgeClass };
