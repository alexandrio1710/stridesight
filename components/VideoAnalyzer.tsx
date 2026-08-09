'use client';

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type ReactNode,
} from 'react';
import type { Pose as PoseInstance, Results } from '@mediapipe/pose';
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileVideo,
  Info,
  Loader2,
  RotateCcw,
  Target,
  UploadCloud,
} from 'lucide-react';
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
  generateSessionNarrative,
  getSideJoints,
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
  MIN_GAIT_EVENTS_FOR_SUMMARY,
  MIN_SAMPLES_FOR_SUMMARY,
  MIN_STEPS_FOR_CADENCE,
  PHASE_DESCRIPTIONS,
  PHASE_LABELS,
  PHASE_THRESHOLDS,
  POSE_LANDMARK_INDICES,
  SPRINT_PHASES,
  TOE_OFF_SEARCH_WINDOW_SECONDS,
  TRUNK_LEAN_PHASE_TIME_CONSTANT_SECONDS,
  type AngleSample,
  type FrameMetrics,
  type Landmark,
  type MetricStatus,
  type PhaseAggregates,
  type PoseLandmarks,
  type SessionSummary,
  type Side,
  type SprintPhase,
} from '@/utils/biomechanics';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

type ModelState = 'idle' | 'loading' | 'ready' | 'error';
type PhaseSelection = 'auto' | SprintPhase;

const MEDIAPIPE_CDN_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/pose';
const ACCEPTED_MIME_TYPES = ['video/mp4', 'video/webm', 'video/quicktime'];
const ACCEPTED_EXTENSION_PATTERN = /\.(mp4|webm|mov)$/i;

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
  phaseAggregates: PhaseAggregates
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
      const strideAggregates = phaseAggregates[state.phaseAtGroundContact ?? currentPhase];
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

/** [startLandmarkIndex, endLandmarkIndex] pair, as exported by @mediapipe/pose's POSE_CONNECTIONS. */
type Connection = readonly [number, number];

// ---------------------------------------------------------------------------
// Canvas drawing helpers (module-level — pure functions of ctx + data)
// ---------------------------------------------------------------------------

function statusToRGBA(status: MetricStatus | null, alpha: number): string {
  switch (status) {
    case 'optimal':
      return `rgba(74, 222, 128, ${alpha})`; // green-400
    case 'caution':
      return `rgba(250, 204, 21, ${alpha})`; // yellow-400
    case 'suboptimal':
      return `rgba(248, 113, 113, ${alpha})`; // red-400
    default:
      return `rgba(148, 163, 184, ${alpha})`; // slate-400
  }
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: PoseLandmarks,
  connections: readonly Connection[],
  width: number,
  height: number
): void {
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.85)'; // sky-400

  connections.forEach(([startIdx, endIdx]) => {
    const start = landmarks[startIdx];
    const end = landmarks[endIdx];
    if (!isLandmarkReliable(start) || !isLandmarkReliable(end)) return;

    ctx.beginPath();
    ctx.moveTo(start.x * width, start.y * height);
    ctx.lineTo(end.x * width, end.y * height);
    ctx.stroke();
  });

  ctx.fillStyle = 'rgba(224, 242, 254, 0.95)'; // sky-100
  landmarks.forEach((landmark) => {
    if (!isLandmarkReliable(landmark)) return;
    ctx.beginPath();
    ctx.arc(landmark.x * width, landmark.y * height, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

/**
 * Draws a semi-transparent pie-slice arc at `vertex`, spanning the interior
 * angle between rays vertex->a and vertex->c. The sweep direction is chosen
 * so the arc always fills the *shorter* (interior) angle rather than the
 * reflex angle, matching the value produced by calculateAngle().
 */
function drawJointArc(
  ctx: CanvasRenderingContext2D,
  a: Landmark,
  vertex: Landmark,
  c: Landmark,
  width: number,
  height: number,
  color: string,
  radius = 32
): void {
  const pVertex = { x: vertex.x * width, y: vertex.y * height };
  const pA = { x: a.x * width, y: a.y * height };
  const pC = { x: c.x * width, y: c.y * height };

  const angleA = Math.atan2(pA.y - pVertex.y, pA.x - pVertex.x);
  const angleC = Math.atan2(pC.y - pVertex.y, pC.x - pVertex.x);

  let delta = angleC - angleA;
  while (delta > Math.PI) delta -= 2 * Math.PI;
  while (delta < -Math.PI) delta += 2 * Math.PI;

  const anticlockwise = delta < 0;

  ctx.beginPath();
  ctx.moveTo(pVertex.x, pVertex.y);
  ctx.arc(pVertex.x, pVertex.y, radius, angleA, angleC, anticlockwise);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();

  ctx.lineWidth = 1.5;
  ctx.strokeStyle = color.replace(/, [\d.]+\)$/, ', 0.9)');
  ctx.stroke();
}

/**
 * Draws the trunk-lean arc at the hip, between the torso line and a vertical
 * reference — the visual counterpart to the trunk lean angle that drives
 * phase detection. Uses shoulder/hip midpoints computed here (not exported
 * from the biomechanics module, which only needs the final angle).
 */
function drawTrunkLeanArc(
  ctx: CanvasRenderingContext2D,
  landmarks: PoseLandmarks,
  metrics: FrameMetrics,
  width: number,
  height: number
): void {
  if (metrics.trunkLeanAngle === null) return;

  const idx = POSE_LANDMARK_INDICES;
  const leftShoulder = landmarks[idx.LEFT_SHOULDER];
  const rightShoulder = landmarks[idx.RIGHT_SHOULDER];
  const leftHip = landmarks[idx.LEFT_HIP];
  const rightHip = landmarks[idx.RIGHT_HIP];

  if (
    !isLandmarkReliable(leftShoulder) ||
    !isLandmarkReliable(rightShoulder) ||
    !isLandmarkReliable(leftHip) ||
    !isLandmarkReliable(rightHip)
  ) {
    return;
  }

  const shoulderMid: Landmark = {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2,
    z: 0,
  };
  const hipMid: Landmark = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2,
    z: 0,
  };
  const verticalReference: Landmark = { x: hipMid.x, y: hipMid.y - 0.15, z: 0 };

  drawJointArc(
    ctx,
    shoulderMid,
    hipMid,
    verticalReference,
    width,
    height,
    statusToRGBA(metrics.trunkLeanStatus, 0.3),
    46
  );
}

function drawAngleArcs(
  ctx: CanvasRenderingContext2D,
  landmarks: PoseLandmarks,
  metrics: FrameMetrics,
  width: number,
  height: number
): void {
  drawTrunkLeanArc(ctx, landmarks, metrics, width, height);

  if (metrics.kneeDriveAngle !== null && metrics.kneeDriveSide) {
    const { hip, knee, ankle } = getSideJoints(landmarks, metrics.kneeDriveSide);
    drawJointArc(ctx, hip, knee, ankle, width, height, statusToRGBA(metrics.kneeDriveStatus, 0.35));
  }

  if (metrics.hipExtensionAngle !== null && metrics.hipExtensionSide) {
    const { shoulder, hip, knee } = getSideJoints(landmarks, metrics.hipExtensionSide);
    drawJointArc(ctx, shoulder, hip, knee, width, height, statusToRGBA(metrics.hipExtensionStatus, 0.35));
  }

  if (metrics.leftArmSwingAngle !== null) {
    const { shoulder, elbow, wrist } = getSideJoints(landmarks, 'left');
    drawJointArc(ctx, shoulder, elbow, wrist, width, height, statusToRGBA(metrics.leftArmSwingStatus, 0.35), 24);
  }

  if (metrics.rightArmSwingAngle !== null) {
    const { shoulder, elbow, wrist } = getSideJoints(landmarks, 'right');
    drawJointArc(ctx, shoulder, elbow, wrist, width, height, statusToRGBA(metrics.rightArmSwingStatus, 0.35), 24);
  }
}

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
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
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
  const [isDragging, setIsDragging] = useState(false);
  const [liveMetrics, setLiveMetrics] = useState<FrameMetrics | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [isTrackingLost, setIsTrackingLost] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  // Distinguishes "hasn't pressed play yet" from "played the whole clip and
  // got zero usable frames" -- both look identical as frameCount === 0 &&
  // !isAnalyzing, but they call for very different messaging (see the empty
  // Phase Reports state below).
  const [hasAttemptedPlayback, setHasAttemptedPlayback] = useState(false);
  const [phaseSummaries, setPhaseSummaries] = useState<Partial<Record<SprintPhase, SessionSummary>>>({});
  const [phaseOverride, setPhaseOverride] = useState<PhaseSelection>('auto');

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

        if (
          buffer.length === 3 &&
          isStepPeak(
            [buffer[0], buffer[1], buffer[2]],
            lastStepPeakTimestampRef.current
          )
        ) {
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
        phaseAggregatesRef.current
      );
      processGaitEvents(
        gaitEventStateRef.current.right,
        'right',
        isLandmarkReliable(rightAnkle) ? rightAnkle.y : null,
        metrics.rightKneeAngle,
        metrics.timestampSeconds,
        worldLandmarks,
        phase,
        phaseAggregatesRef.current
      );

      setLiveMetrics(smoothDisplayMetrics(displayEmaRef.current, metrics));
      setFrameCount(frameHistoryRef.current.length);
      setPhaseSummaries(computePhaseSummaries(phaseAggregatesRef.current));

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

  // -- File handling -----------------------------------------------------

  const handleFile = useCallback((file: File) => {
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
    displayEmaRef.current = createDisplayEma();

    setErrorMessage(null);
    setLiveMetrics(null);
    setFrameCount(0);
    setIsTrackingLost(false);
    setIsAnalyzing(false);
    setHasAttemptedPlayback(false);
    setPhaseSummaries({});
    setFileName(file.name);
    setVideoSrc(url);

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsDragging(false);
      const file = event.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
  }, []);

  const onFileInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) handleFile(file);
      event.target.value = '';
    },
    [handleFile]
  );

  const handleReset = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
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
    displayEmaRef.current = createDisplayEma();

    setVideoSrc(null);
    setFileName('');
    setLiveMetrics(null);
    setFrameCount(0);
    setErrorMessage(null);
    setIsTrackingLost(false);
    setIsAnalyzing(false);
    setHasAttemptedPlayback(false);
    setPhaseSummaries({});
  }, []);

  const handleDownloadCSV = useCallback(() => {
    if (frameHistoryRef.current.length === 0) return;
    const csv = generateCSV(frameHistoryRef.current, phaseSummaries);
    const safeName = fileName.replace(/\.[^/.]+$/, '') || 'stridesight-analysis';
    downloadCSV(`${safeName}-angles.csv`, csv);
  }, [fileName, phaseSummaries]);

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
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleReset}
                className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-sm font-medium text-slate-300 transition-colors hover:bg-white/10"
              >
                <RotateCcw className="h-4 w-4" />
                New Video
              </button>
              <button
                type="button"
                onClick={handleDownloadCSV}
                disabled={frameCount === 0}
                className="inline-flex items-center gap-1.5 rounded-lg bg-sky-500 px-3 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
              >
                <Download className="h-4 w-4" />
                Download CSV
              </button>
            </div>
          </div>

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
