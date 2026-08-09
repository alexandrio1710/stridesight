/**
 * StrideSight — Biomechanics Engine
 *
 * Pure, framework-agnostic math for converting MediaPipe Pose landmarks into
 * sprint-mechanics joint angles (via 3D vector dot products), classifying a
 * runner's current sprint phase, scoring those angles against phase-specific
 * coaching thresholds, and serializing results to CSV.
 *
 * 2D vs. 3D — and why it's a deliberate split, not a blanket upgrade:
 *
 * MediaPipe Pose emits two parallel landmark streams per frame: `poseLandmarks`
 * (normalized image-plane coordinates, x/y in [0,1] plus a rough relative z)
 * and `poseWorldLandmarks` (real-world 3D coordinates in meters, origin at the
 * hip midpoint). Knee drive, hip extension, and arm swing are angles *between
 * body segments* — a 2D image-plane projection foreshortens them whenever the
 * limb's plane of motion isn't exactly perpendicular to the camera (a runner
 * angled slightly toward the lens, an arm swinging partly toward/away from
 * it), so those three are computed from the 3D world landmarks.
 *
 * Trunk lean is different: it's an angle measured *against true vertical*,
 * not just between two body segments, and neither Google's documentation nor
 * the MediaPipe source publicly specifies whether `poseWorldLandmarks`' axes
 * are gravity-aligned or merely camera-relative — it's a pure vision model
 * with no gravity sensor, so there's no guarantee "up" in that 3D space means
 * anything physical. Silently assuming an unverified axis convention would
 * risk inverting or skewing every trunk-lean reading. The 2D image y-axis
 * (increasing downward), by contrast, is an unambiguous, documented
 * convention — and camera-level footage makes it a fine proxy for vertical,
 * which is the same assumption an unverified 3D axis would need anyway. So
 * trunk lean stays a 2D image-plane calculation on purpose; see
 * calculateTrunkLeanAngle() below.
 *
 * Why phase-specific thresholds exist at all:
 *
 * Acceleration and max-velocity sprinting are biomechanically different
 * gaits, not the same gait done "better" or "worse." A single fixed set of
 * "good angle" thresholds applied uniformly across a whole sprint will
 * misjudge correct technique in whichever phase wasn't the reference —
 * e.g. penalizing normal low-to-the-ground drive-phase strides for not
 * having max-velocity knee lift, or penalizing max-velocity strides for not
 * extending the hip as far as the drive phase does. This module classifies
 * each frame into a sprint phase (from trunk lean, the textbook signal for
 * this) and scores it against that phase's own targets.
 *
 * The phase thresholds below are synthesized from published sprint
 * biomechanics literature (see the project's chat history / commit message
 * for citations), not from a trained model — MediaPipe's pose detector is
 * the only pretrained model in this pipeline, and everything downstream is
 * a rules-based coaching heuristic. Treat scores as a coaching
 * conversation-starter: technique varies by sprinter body type, event, and
 * training background, and even elite sprinters don't share one "correct"
 * number for every joint.
 *
 * No DOM, no React, no MediaPipe imports — fully unit-testable in isolation.
 */

// ---------------------------------------------------------------------------
// Core geometric types
// ---------------------------------------------------------------------------

export interface Point2D {
  x: number;
  y: number;
}

export interface Point3D extends Point2D {
  z: number;
}

/**
 * Structurally compatible with both of MediaPipe's landmark shapes:
 * `NormalizedLandmark` (image-plane, x/y in [0,1]) and `Landmark` (world,
 * real-world meters). Same fields either way — only the coordinate frame
 * differs, which is why one `PoseLandmarks` type serves both call sites
 * throughout this file (see the 2D-vs-3D note at the top of the file).
 */
export interface Landmark extends Point3D {
  visibility?: number;
}

export type PoseLandmarks = Landmark[];

export type Side = 'left' | 'right';

export type MetricStatus = 'optimal' | 'caution' | 'suboptimal';

/**
 * The three phases sprint coaching literature distinguishes by posture and
 * force application. "Acceleration" covers the block start through the
 * drive phase; "transition" is the real, named phase where the sprinter
 * rises from the drive position toward upright; "maxVelocity" is top-speed
 * running.
 */
export type SprintPhase = 'acceleration' | 'transition' | 'maxVelocity';

export const SPRINT_PHASES: readonly SprintPhase[] = ['acceleration', 'transition', 'maxVelocity'];

export const PHASE_LABELS: Record<SprintPhase, string> = {
  acceleration: 'Acceleration',
  transition: 'Transition',
  maxVelocity: 'Max Velocity',
};

export const PHASE_DESCRIPTIONS: Record<SprintPhase, string> = {
  acceleration: 'Driving out of the start — forward trunk lean, applying horizontal force.',
  transition: 'Rising out of the drive phase toward a tall, upright sprinting posture.',
  maxVelocity: 'Top-speed running — tall posture, fast front-side knee drive.',
};

// ---------------------------------------------------------------------------
// MediaPipe BlazePose landmark topology (33-point model)
// https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
// ---------------------------------------------------------------------------

export const POSE_LANDMARK_INDICES = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
} as const;

/** Landmarks below this visibility confidence are treated as untracked. */
export const MIN_VISIBILITY_THRESHOLD = 0.5;

export function isLandmarkReliable(landmark: Landmark | undefined | null): landmark is Landmark {
  if (!landmark) return false;
  if (typeof landmark.visibility !== 'number') return true;
  return landmark.visibility >= MIN_VISIBILITY_THRESHOLD;
}

interface SideJoints {
  shoulder: Landmark;
  hip: Landmark;
  knee: Landmark;
  ankle: Landmark;
  elbow: Landmark;
  wrist: Landmark;
}

/** Look up the six joints (shoulder/hip/knee/ankle/elbow/wrist) for one body side. */
export function getSideJoints(landmarks: PoseLandmarks, side: Side): SideJoints {
  const idx = POSE_LANDMARK_INDICES;
  if (side === 'left') {
    return {
      shoulder: landmarks[idx.LEFT_SHOULDER],
      hip: landmarks[idx.LEFT_HIP],
      knee: landmarks[idx.LEFT_KNEE],
      ankle: landmarks[idx.LEFT_ANKLE],
      elbow: landmarks[idx.LEFT_ELBOW],
      wrist: landmarks[idx.LEFT_WRIST],
    };
  }
  return {
    shoulder: landmarks[idx.RIGHT_SHOULDER],
    hip: landmarks[idx.RIGHT_HIP],
    knee: landmarks[idx.RIGHT_KNEE],
    ankle: landmarks[idx.RIGHT_ANKLE],
    elbow: landmarks[idx.RIGHT_ELBOW],
    wrist: landmarks[idx.RIGHT_WRIST],
  };
}

// ---------------------------------------------------------------------------
// Vector mathematics
// ---------------------------------------------------------------------------

/**
 * Computes the interior angle ABC (in degrees) at vertex `b`, formed by rays
 * b->a and b->c, using the vector dot product identity:
 *
 *   cos(theta) = (BA . BC) / (|BA| * |BC|)
 *
 * Takes full 3D points — the dot product and magnitude both extend to 3D
 * without changing shape (magnitude via Math.hypot(x, y, z)), so this same
 * function serves both a true 3D angle (world landmarks, z meaningful) and a
 * 2D-in-3D-clothing angle (z fixed at 0 for all three points, which the
 * algebra reduces to exactly the 2D formula) — see calculateTrunkLeanAngle().
 *
 * The cosine is clamped to [-1, 1] to guard against floating-point drift
 * (e.g. -1.0000000002) that would otherwise make Math.acos return NaN.
 */
export function calculateAngle(a: Point3D, b: Point3D, c: Point3D): number {
  const vectorBA: Point3D = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const vectorBC: Point3D = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };

  const dotProduct = vectorBA.x * vectorBC.x + vectorBA.y * vectorBC.y + vectorBA.z * vectorBC.z;
  const magnitudeBA = Math.hypot(vectorBA.x, vectorBA.y, vectorBA.z);
  const magnitudeBC = Math.hypot(vectorBC.x, vectorBC.y, vectorBC.z);

  if (magnitudeBA === 0 || magnitudeBC === 0) {
    return 0;
  }

  const cosTheta = dotProduct / (magnitudeBA * magnitudeBC);
  const clampedCosTheta = Math.min(1, Math.max(-1, cosTheta));
  const radians = Math.acos(clampedCosTheta);

  return radians * (180 / Math.PI);
}

export interface LegRoles {
  leadSide: Side;
  trailSide: Side;
}

/**
 * Identifies which leg is currently "leading" (driving forward/upward through
 * flexion) versus "trailing" (extended behind the body) for the current
 * frame, using knee height as a proxy.
 *
 * In normalized image coordinates y grows downward, so the knee with the
 * smaller y value is physically higher and is treated as the lead knee.
 * Returns null if either knee isn't confidently tracked.
 */
export function determineLegRoles(landmarks: PoseLandmarks): LegRoles | null {
  const leftKnee = landmarks[POSE_LANDMARK_INDICES.LEFT_KNEE];
  const rightKnee = landmarks[POSE_LANDMARK_INDICES.RIGHT_KNEE];

  if (!isLandmarkReliable(leftKnee) || !isLandmarkReliable(rightKnee)) {
    return null;
  }

  if (leftKnee.y < rightKnee.y) {
    return { leadSide: 'left', trailSide: 'right' };
  }
  return { leadSide: 'right', trailSide: 'left' };
}

function midpoint(a: Point2D, b: Point2D): Point2D {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

interface TorsoMidpoints {
  shoulderMid: Point2D;
  hipMid: Point2D;
}

/**
 * Averages both shoulders and both hips into torso midpoints, rather than
 * using one body side. That makes trunk lean robust to the runner being
 * photographed slightly off-axis, and independent of which leg currently
 * leads.
 */
function getTorsoMidpoints(landmarks: PoseLandmarks): TorsoMidpoints | null {
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
    return null;
  }

  return {
    shoulderMid: midpoint(leftShoulder, rightShoulder),
    hipMid: midpoint(leftHip, rightHip),
  };
}

/**
 * Forward trunk lean: the angle between the torso (hip-to-shoulder line) and
 * true vertical, measured at the hip. 0° is a perfectly upright torso;
 * larger values mean more forward (or backward) lean.
 *
 * This is the primary signal sprint biomechanics research uses to
 * distinguish acceleration from max-velocity running: elite sprinters lean
 * roughly 40-50° at block exit and progressively straighten to
 * near-vertical by top speed.
 *
 * Deliberately a 2D image-plane calculation, not 3D — see the file-level
 * comment on 2D vs. 3D for why. Takes `landmarks` (image-plane), never
 * `poseWorldLandmarks`. The z coordinate is pinned to 0 on all three points
 * passed to calculateAngle(), which algebraically collapses its 3D formula
 * back to the plain 2D one — this is a real 2D calculation wearing a 3D
 * function signature, not an accidental 3D one.
 *
 * Only meaningful with a roughly side-on (lateral) camera angle and a
 * level (non-tilted) camera — filmed head-on, from behind, or with the
 * camera itself rotated, the image y-axis no longer approximates true
 * vertical.
 */
export function calculateTrunkLeanAngle(landmarks: PoseLandmarks): number | null {
  const torso = getTorsoMidpoints(landmarks);
  if (!torso) return null;

  const shoulderMid: Point3D = { x: torso.shoulderMid.x, y: torso.shoulderMid.y, z: 0 };
  const hipMid: Point3D = { x: torso.hipMid.x, y: torso.hipMid.y, z: 0 };
  // A point directly "above" the hip in image space represents true vertical.
  // calculateAngle only uses this point's direction from the hip, not the
  // magnitude of the offset, so an arbitrary offset of 1 is sufficient.
  const verticalReference: Point3D = { x: hipMid.x, y: hipMid.y - 1, z: 0 };

  return calculateAngle(shoulderMid, hipMid, verticalReference);
}

/**
 * Exponential moving average with a fixed per-call weight. Appropriate for
 * a purely cosmetic smoothing pass that's meant to soften jitter *between
 * calls*, not to hold to any particular real-time responsiveness — see
 * smoothDisplayMetrics() in the component, its one caller.
 *
 * Deliberately NOT used for sprint-phase classification — see
 * smoothValueOverTime() below for why a fixed per-call alpha is the wrong
 * tool for that.
 */
export function smoothValue(previous: number | null, next: number, alpha = 0.15): number {
  if (previous === null) return next;
  return alpha * next + (1 - alpha) * previous;
}

/**
 * Time-aware exponential moving average: converges toward `next` on a fixed
 * *real-time* schedule (`timeConstantSeconds`), regardless of how often this
 * is called. Equivalent to smoothValue() with alpha recomputed each call as
 * `1 - exp(-dtSeconds / timeConstantSeconds)`, the standard continuous-time
 * EMA formula.
 *
 * This distinction matters for sprint-phase classification specifically:
 * MediaPipe pose inference runs at a rate that varies by device, browser,
 * and momentary CPU load, not at a fixed frame rate. A plain per-call alpha
 * (smoothValue()) implicitly assumes a roughly constant call rate — its
 * *real-time* responsiveness scales with however often it happens to get
 * called, so the same alpha can be heavily smoothed on a fast device and
 * barely smoothed at all on a slower one processing fewer frames per
 * second. That mismatch previously let ordinary stride-to-stride trunk-lean
 * noise (arm swing rotates the shoulders slightly with every step) cross a
 * phase boundary and get reported as a real phase change, splitting one
 * genuinely single-phase clip into two reported phases. Pinning the
 * smoothing to a real time constant instead — long enough to average out a
 * stride cycle (~0.3-0.4s at max velocity), short enough to still track a
 * genuine multi-second phase transition — makes classification consistent
 * regardless of inference throughput.
 */
export function smoothValueOverTime(
  previous: number | null,
  next: number,
  dtSeconds: number,
  timeConstantSeconds: number
): number {
  if (previous === null) return next;
  if (dtSeconds <= 0) return previous;
  const alpha = 1 - Math.exp(-dtSeconds / timeConstantSeconds);
  return alpha * next + (1 - alpha) * previous;
}

// ---------------------------------------------------------------------------
// Phase-specific coaching thresholds
// ---------------------------------------------------------------------------

interface LowerBoundThreshold {
  optimalMin: number;
  cautionMin: number;
}

interface UpperBoundThreshold {
  optimalMax: number;
  cautionMax: number;
}

interface TargetRangeThreshold {
  optimalMin: number;
  optimalMax: number;
  cautionMin: number;
  cautionMax: number;
}

export interface PhaseThresholdSet {
  /** Angle from vertical at the hip. Too little OR too much lean is a fault. */
  trunkLean: TargetRangeThreshold;
  /** Hip-Knee-Ankle angle of the lead leg. Higher = more aggressive knee drive. */
  kneeDrive: LowerBoundThreshold;
  /** Shoulder-Hip-Knee angle of the trail leg. Higher = fuller hip extension. */
  hipExtension: LowerBoundThreshold;
  /** Shoulder-Elbow-Wrist angle. */
  armSwing: TargetRangeThreshold;
  /** Hip-Knee-Ankle angle of the stance leg, sampled at its detected toe-off instant. */
  kneeExtensionAtToeOff: LowerBoundThreshold;
  /** Seconds between a detected ground contact and that same leg's detected toe-off. */
  groundContactTime: UpperBoundThreshold;
}

/**
 * Phase-specific coaching targets — deliberately different per phase, not
 * the same numbers reused three times:
 *
 *  - Trunk lean: large during acceleration (~30-60° at block exit per
 *    published kinematics), falling through a narrow transition band, to
 *    near-vertical (0-10°) at max velocity. Max velocity has no lower bound
 *    on "good" — you cannot be too upright at top speed.
 *  - Knee drive: front-side mechanics are the defining hallmark of elite
 *    MAX-VELOCITY technique, so the bar is highest there. Early
 *    acceleration strides are naturally lower to the ground; judging them
 *    against max-velocity knee lift would flag correct technique as wrong.
 *  - Hip extension: horizontal force demand is greatest during
 *    acceleration, so aggressive extension is rewarded most there. Research
 *    on elite sprinters shows they terminate ground contact BEFORE reaching
 *    full hip/knee extension at max velocity, favoring fast leg recovery
 *    over maximal extension — so the max-velocity target is intentionally
 *    less extreme than acceleration's, not more.
 *  - Arm swing: naturally larger amplitude during the drive phase (bigger,
 *    more powerful arm action) and tightens to a compact ~90° cycle by max
 *    velocity.
 *  - Knee extension at toe-off: the same "elite sprinters terminate ground
 *    contact before reaching full extension at max velocity" research that
 *    shapes hip extension applies at the knee too — so, like hip extension,
 *    the max-velocity target is intentionally the least extreme of the
 *    three phases, not the most.
 *  - Ground contact time: the mechanical cause behind that same "quick,
 *    elastic contact" pattern. Published sprint-kinematics data puts elite
 *    ground contact times at roughly 80-100ms at max velocity, versus a
 *    noticeably longer first-few-steps contact during acceleration (driving
 *    against the ground to build horizontal force simply takes more time
 *    than a top-speed stride does) — so unlike the other four metrics, this
 *    one gets progressively *stricter* (shorter) target ceilings from
 *    acceleration through max velocity, not looser.
 */
export const PHASE_THRESHOLDS: Record<SprintPhase, PhaseThresholdSet> = {
  acceleration: {
    trunkLean: { optimalMin: 32, optimalMax: 58, cautionMin: 28, cautionMax: 68 },
    kneeDrive: { optimalMin: 72, cautionMin: 58 },
    hipExtension: { optimalMin: 155, cautionMin: 135 },
    armSwing: { optimalMin: 65, optimalMax: 115, cautionMin: 45, cautionMax: 135 },
    kneeExtensionAtToeOff: { optimalMin: 162, cautionMin: 145 },
    groundContactTime: { optimalMax: 0.18, cautionMax: 0.24 },
  },
  transition: {
    trunkLean: { optimalMin: 13, optimalMax: 28, cautionMin: 8, cautionMax: 33 },
    kneeDrive: { optimalMin: 80, cautionMin: 65 },
    hipExtension: { optimalMin: 150, cautionMin: 130 },
    armSwing: { optimalMin: 72, optimalMax: 108, cautionMin: 52, cautionMax: 128 },
    kneeExtensionAtToeOff: { optimalMin: 158, cautionMin: 140 },
    groundContactTime: { optimalMax: 0.14, cautionMax: 0.19 },
  },
  maxVelocity: {
    trunkLean: { optimalMin: 0, optimalMax: 10, cautionMin: 0, cautionMax: 13 },
    kneeDrive: { optimalMin: 90, cautionMin: 75 },
    hipExtension: { optimalMin: 140, cautionMin: 120 },
    armSwing: { optimalMin: 80, optimalMax: 100, cautionMin: 60, cautionMax: 120 },
    kneeExtensionAtToeOff: { optimalMin: 150, cautionMin: 135 },
    groundContactTime: { optimalMax: 0.1, cautionMax: 0.14 },
  },
} as const;

/**
 * Over-stride distance (hip-to-ankle horizontal gap at ground contact, in
 * meters) uses the same fault regardless of sprint phase — landing ahead of
 * the center of mass creates a braking force whenever it happens — so
 * unlike the angle metrics above, one threshold pair covers all three
 * phases rather than being duplicated per phase.
 */
const OVERSTRIDE_CAUTION_M = 0.2;
const OVERSTRIDE_SUBOPTIMAL_M = 0.3;

/**
 * Classifies sprint phase from smoothed trunk lean. The boundaries are the
 * acceleration/max-velocity caution edges from PHASE_THRESHOLDS above (not a
 * separately-tuned constant), so classification and scoring can never drift
 * out of sync with each other.
 */
export function classifyPhase(smoothedLeanAngle: number): SprintPhase {
  if (smoothedLeanAngle >= PHASE_THRESHOLDS.acceleration.trunkLean.cautionMin) return 'acceleration';
  if (smoothedLeanAngle <= PHASE_THRESHOLDS.maxVelocity.trunkLean.cautionMax) return 'maxVelocity';
  return 'transition';
}

/** Degrees of "dead zone" past a phase boundary required before crossing it, in either direction. */
const PHASE_HYSTERESIS_DEG = 3;

/**
 * Real-time constant (seconds) for smoothing trunk lean before phase
 * classification — see smoothValueOverTime() for why this must be a real
 * time constant rather than a fixed per-call alpha. Long enough to average
 * out a full stride cycle (~0.3-0.4s at max velocity, where arm swing
 * couples a few degrees of rotation into the shoulders every step) without
 * being so long it blurs out a genuine multi-second phase transition.
 */
export const TRUNK_LEAN_PHASE_TIME_CONSTANT_SECONDS = 0.8;

/**
 * classifyPhase() alone can flicker if smoothed lean hovers right at a
 * boundary for an extended stretch (e.g. a sprinter easing out of the drive
 * phase very gradually) — even EMA smoothing only damps high-frequency
 * noise, it doesn't stop a signal that's genuinely sitting on a threshold.
 *
 * This wraps classifyPhase with a full Schmitt-trigger dead zone at *both*
 * boundaries, in *both* directions: leaving a phase requires crossing
 * meaningfully past its own boundary, and — importantly — so does entering
 * a neighboring phase from 'transition'. An earlier version only damped the
 * "leaving acceleration/maxVelocity" direction on the (untested) assumption
 * that bouncing only ran one way; real footage of a clip that was entirely
 * one true phase throughout showed the opposite also happens (repeatedly
 * re-entering maxVelocity from transition, undamped, then immediately
 * bouncing back out), splitting a single real phase into two reported ones.
 * Damping all four crossings closes that gap.
 */
export function classifyPhaseWithHysteresis(
  smoothedLeanAngle: number,
  previousPhase: SprintPhase | null
): SprintPhase {
  const accelBoundary = PHASE_THRESHOLDS.acceleration.trunkLean.cautionMin;
  const maxVelBoundary = PHASE_THRESHOLDS.maxVelocity.trunkLean.cautionMax;

  if (previousPhase === 'acceleration') {
    if (smoothedLeanAngle >= accelBoundary - PHASE_HYSTERESIS_DEG) return 'acceleration';
    return classifyPhase(smoothedLeanAngle);
  }
  if (previousPhase === 'maxVelocity') {
    if (smoothedLeanAngle <= maxVelBoundary + PHASE_HYSTERESIS_DEG) return 'maxVelocity';
    return classifyPhase(smoothedLeanAngle);
  }
  if (previousPhase === 'transition') {
    if (smoothedLeanAngle >= accelBoundary + PHASE_HYSTERESIS_DEG) return 'acceleration';
    if (smoothedLeanAngle <= maxVelBoundary - PHASE_HYSTERESIS_DEG) return 'maxVelocity';
    return 'transition';
  }
  return classifyPhase(smoothedLeanAngle);
}

/** Classifies a "higher is better" metric (knee drive, hip extension). */
export function classifyByLowerBound(
  angle: number,
  optimalMin: number,
  cautionMin: number
): MetricStatus {
  if (angle > optimalMin) return 'optimal';
  if (angle > cautionMin) return 'caution';
  return 'suboptimal';
}

/** Classifies a "closer to a target band is better" metric (trunk lean, arm swing). */
export function classifyByTargetRange(
  angle: number,
  optimalMin: number,
  optimalMax: number,
  cautionMin: number,
  cautionMax: number
): MetricStatus {
  if (angle >= optimalMin && angle <= optimalMax) return 'optimal';
  if (angle >= cautionMin && angle <= cautionMax) return 'caution';
  return 'suboptimal';
}

// ---------------------------------------------------------------------------
// Frame-level metric computation
// ---------------------------------------------------------------------------

export interface FrameMetrics {
  frameIndex: number;
  timestampSeconds: number;
  phase: SprintPhase;

  trunkLeanAngle: number | null;
  trunkLeanStatus: MetricStatus | null;

  kneeDriveAngle: number | null;
  kneeDriveSide: Side | null;
  kneeDriveStatus: MetricStatus | null;

  hipExtensionAngle: number | null;
  hipExtensionSide: Side | null;
  hipExtensionStatus: MetricStatus | null;

  leftArmSwingAngle: number | null;
  leftArmSwingStatus: MetricStatus | null;

  rightArmSwingAngle: number | null;
  rightArmSwingStatus: MetricStatus | null;

  /**
   * Raw per-side Hip-Knee-Ankle angle (3D world), computed unconditionally
   * for both legs regardless of lead/trail role — unlike kneeDriveAngle
   * (lead leg only), these feed the gait-event detectors (ground contact /
   * toe-off), which need to track each anatomical leg continuously rather
   * than whichever leg currently leads. See recordGaitEvent() in the
   * component and the "Gait event detection" section below.
   */
  leftKneeAngle: number | null;
  rightKneeAngle: number | null;
}

/**
 * Computes every tracked biomechanical angle for a single video frame,
 * scored against the given sprint phase's thresholds. `phase` is supplied
 * by the caller (typically classified from an EMA-smoothed trunk lean
 * maintained across frames) since it depends on temporal state this
 * otherwise-pure function doesn't hold itself.
 *
 * Takes two parallel landmark sets for the same frame:
 *  - `landmarks`: image-plane coordinates (MediaPipe's `poseLandmarks`).
 *    Used for trunk lean (needs true vertical — see the file-level 2D-vs-3D
 *    note) and for deciding which leg currently leads (needs on-screen
 *    height, same reasoning).
 *  - `worldLandmarks`: real-world 3D coordinates in meters, same 33-point
 *    indexing (MediaPipe's `poseWorldLandmarks`). Used for the three angles
 *    that are purely relationships between body segments — knee drive, hip
 *    extension, arm swing — where 3D removes the 2D projection's
 *    foreshortening error.
 *
 * Any joint chain with unreliable (low-visibility/occluded) landmarks is
 * left as `null` rather than reporting a misleading angle.
 */
export function computeFrameMetrics(
  landmarks: PoseLandmarks,
  worldLandmarks: PoseLandmarks,
  frameIndex: number,
  timestampSeconds: number,
  phase: SprintPhase
): FrameMetrics {
  const thresholds = PHASE_THRESHOLDS[phase];

  const trunkLeanAngle = calculateTrunkLeanAngle(landmarks);
  const trunkLeanStatus =
    trunkLeanAngle !== null
      ? classifyByTargetRange(
          trunkLeanAngle,
          thresholds.trunkLean.optimalMin,
          thresholds.trunkLean.optimalMax,
          thresholds.trunkLean.cautionMin,
          thresholds.trunkLean.cautionMax
        )
      : null;

  // Lead/trail role is decided from on-screen (2D) knee height, then the
  // resulting side is looked up in the 3D world landmarks for the actual
  // angle — "which leg" is a 2D/vertical question, "what angle" is a 3D one.
  const legRoles = determineLegRoles(landmarks);

  let kneeDriveAngle: number | null = null;
  let kneeDriveSide: Side | null = null;
  let kneeDriveStatus: MetricStatus | null = null;

  let hipExtensionAngle: number | null = null;
  let hipExtensionSide: Side | null = null;
  let hipExtensionStatus: MetricStatus | null = null;

  if (legRoles) {
    const lead = getSideJoints(worldLandmarks, legRoles.leadSide);
    if (isLandmarkReliable(lead.hip) && isLandmarkReliable(lead.knee) && isLandmarkReliable(lead.ankle)) {
      kneeDriveAngle = calculateAngle(lead.hip, lead.knee, lead.ankle);
      kneeDriveSide = legRoles.leadSide;
      kneeDriveStatus = classifyByLowerBound(
        kneeDriveAngle,
        thresholds.kneeDrive.optimalMin,
        thresholds.kneeDrive.cautionMin
      );
    }

    const trail = getSideJoints(worldLandmarks, legRoles.trailSide);
    if (
      isLandmarkReliable(trail.shoulder) &&
      isLandmarkReliable(trail.hip) &&
      isLandmarkReliable(trail.knee)
    ) {
      hipExtensionAngle = calculateAngle(trail.shoulder, trail.hip, trail.knee);
      hipExtensionSide = legRoles.trailSide;
      hipExtensionStatus = classifyByLowerBound(
        hipExtensionAngle,
        thresholds.hipExtension.optimalMin,
        thresholds.hipExtension.cautionMin
      );
    }
  }

  let leftArmSwingAngle: number | null = null;
  let leftArmSwingStatus: MetricStatus | null = null;
  const leftArm = getSideJoints(worldLandmarks, 'left');
  if (
    isLandmarkReliable(leftArm.shoulder) &&
    isLandmarkReliable(leftArm.elbow) &&
    isLandmarkReliable(leftArm.wrist)
  ) {
    leftArmSwingAngle = calculateAngle(leftArm.shoulder, leftArm.elbow, leftArm.wrist);
    leftArmSwingStatus = classifyByTargetRange(
      leftArmSwingAngle,
      thresholds.armSwing.optimalMin,
      thresholds.armSwing.optimalMax,
      thresholds.armSwing.cautionMin,
      thresholds.armSwing.cautionMax
    );
  }

  let rightArmSwingAngle: number | null = null;
  let rightArmSwingStatus: MetricStatus | null = null;
  const rightArm = getSideJoints(worldLandmarks, 'right');
  if (
    isLandmarkReliable(rightArm.shoulder) &&
    isLandmarkReliable(rightArm.elbow) &&
    isLandmarkReliable(rightArm.wrist)
  ) {
    rightArmSwingAngle = calculateAngle(rightArm.shoulder, rightArm.elbow, rightArm.wrist);
    rightArmSwingStatus = classifyByTargetRange(
      rightArmSwingAngle,
      thresholds.armSwing.optimalMin,
      thresholds.armSwing.optimalMax,
      thresholds.armSwing.cautionMin,
      thresholds.armSwing.cautionMax
    );
  }

  // Unconditional per-side knee angle (not gated by lead/trail role) — see
  // the FrameMetrics field doc for why gait-event detection needs this
  // instead of kneeDriveAngle.
  let leftKneeAngle: number | null = null;
  const leftLeg = getSideJoints(worldLandmarks, 'left');
  if (isLandmarkReliable(leftLeg.hip) && isLandmarkReliable(leftLeg.knee) && isLandmarkReliable(leftLeg.ankle)) {
    leftKneeAngle = calculateAngle(leftLeg.hip, leftLeg.knee, leftLeg.ankle);
  }

  let rightKneeAngle: number | null = null;
  const rightLeg = getSideJoints(worldLandmarks, 'right');
  if (
    isLandmarkReliable(rightLeg.hip) &&
    isLandmarkReliable(rightLeg.knee) &&
    isLandmarkReliable(rightLeg.ankle)
  ) {
    rightKneeAngle = calculateAngle(rightLeg.hip, rightLeg.knee, rightLeg.ankle);
  }

  return {
    frameIndex,
    timestampSeconds,
    phase,
    trunkLeanAngle,
    trunkLeanStatus,
    kneeDriveAngle,
    kneeDriveSide,
    kneeDriveStatus,
    hipExtensionAngle,
    hipExtensionSide,
    hipExtensionStatus,
    leftArmSwingAngle,
    leftArmSwingStatus,
    rightArmSwingAngle,
    rightArmSwingStatus,
    leftKneeAngle,
    rightKneeAngle,
  };
}

/**
 * Horizontal (direction-of-travel) distance, in meters, between the hip
 * center and `side`'s ankle, from 3D world landmarks — the over-striding
 * measurement, sampled by the caller at the exact frame identified as that
 * leg's ground-contact instant (the gap only matters at footstrike; it's
 * large and unremarkable mid-swing). Assumes a side-on camera, so the
 * world-space X axis (camera-relative — see the file-level 2D-vs-3D note)
 * aligns with the direction of travel.
 */
export function calculateOverstrideDistance(worldLandmarks: PoseLandmarks, side: Side): number | null {
  const idx = POSE_LANDMARK_INDICES;
  const leftHip = worldLandmarks[idx.LEFT_HIP];
  const rightHip = worldLandmarks[idx.RIGHT_HIP];
  const ankle = worldLandmarks[side === 'left' ? idx.LEFT_ANKLE : idx.RIGHT_ANKLE];

  if (!isLandmarkReliable(leftHip) || !isLandmarkReliable(rightHip) || !isLandmarkReliable(ankle)) {
    return null;
  }

  const hipCenterX = (leftHip.x + rightHip.x) / 2;
  return Math.abs(ankle.x - hipCenterX);
}

// ---------------------------------------------------------------------------
// Running aggregation
//
// Per-frame status (optimal/caution/suboptimal) is a coarse 3-bucket signal,
// good for a live overlay but too lossy for a phase-level verdict. These
// aggregates track sum/sum-of-squares incrementally (O(1) per frame) so a
// continuous score and stride-to-stride variability can be derived at any
// point during playback without ever re-scanning the full frame history.
// ---------------------------------------------------------------------------

export interface MetricAggregate {
  count: number;
  sum: number;
  sumOfSquares: number;
}

export function createMetricAggregate(): MetricAggregate {
  return { count: 0, sum: 0, sumOfSquares: 0 };
}

/**
 * Mutates `aggregate` in place. This is the one deliberately non-pure corner
 * of this module: it's called once per tracked metric on every analyzed
 * video frame (so up to a few hundred times a second across five metrics),
 * and it lives in a React ref rather than state, so there's no immutability
 * benefit to buy — only allocation churn to avoid.
 */
export function addSampleInPlace(aggregate: MetricAggregate, value: number): void {
  aggregate.count += 1;
  aggregate.sum += value;
  aggregate.sumOfSquares += value * value;
}

export function aggregateMean(aggregate: MetricAggregate): number | null {
  return aggregate.count > 0 ? aggregate.sum / aggregate.count : null;
}

/** Population standard deviation. Requires at least 2 samples to be meaningful. */
export function aggregateStdDev(aggregate: MetricAggregate): number | null {
  if (aggregate.count < 2) return null;
  const mean = aggregate.sum / aggregate.count;
  const variance = aggregate.sumOfSquares / aggregate.count - mean * mean;
  // Clamp at 0 to guard against a tiny negative from floating-point cancellation.
  return Math.sqrt(Math.max(0, variance));
}

export interface SessionAggregates {
  trunkLean: MetricAggregate;
  kneeDrive: MetricAggregate;
  /** Same knee-drive angle, split by which leg was leading — for a symmetry check only. */
  kneeDriveLeft: MetricAggregate;
  kneeDriveRight: MetricAggregate;
  hipExtension: MetricAggregate;
  leftArmSwing: MetricAggregate;
  rightArmSwing: MetricAggregate;
  /** Step-to-step interval frequency (Hz), one sample per detected step — see isStepPeak(). */
  stepFrequency: MetricAggregate;
  /** Stance-leg Hip-Knee-Ankle angle, one sample per detected toe-off event. */
  kneeExtensionAtToeOff: MetricAggregate;
  /** Hip-to-ankle horizontal gap (meters), one sample per detected ground-contact event. */
  overstride: MetricAggregate;
  /** Seconds from a leg's ground contact to that same leg's toe-off, one sample per completed stance phase. */
  groundContactTime: MetricAggregate;
  /** Seconds from a leg's toe-off to that same leg's next ground contact, one sample per completed swing phase. */
  flightTime: MetricAggregate;
}

export function createSessionAggregates(): SessionAggregates {
  return {
    trunkLean: createMetricAggregate(),
    kneeDrive: createMetricAggregate(),
    kneeDriveLeft: createMetricAggregate(),
    kneeDriveRight: createMetricAggregate(),
    hipExtension: createMetricAggregate(),
    leftArmSwing: createMetricAggregate(),
    rightArmSwing: createMetricAggregate(),
    stepFrequency: createMetricAggregate(),
    kneeExtensionAtToeOff: createMetricAggregate(),
    overstride: createMetricAggregate(),
    groundContactTime: createMetricAggregate(),
    flightTime: createMetricAggregate(),
  };
}

// ---------------------------------------------------------------------------
// Step cadence detection
//
// The knee-drive angle time series naturally oscillates once per step (each
// leg's drive peak alternates with the other leg's), so a local-maximum
// detector on that signal gives a real, causal (online) way to measure step
// frequency without any additional tracking. This is deliberately reported
// as an *informational* metric rather than graded optimal/caution/suboptimal
// like the angle metrics: published research on elite sprinters explicitly
// finds some are "step-frequency reliant" and others "step-length reliant"
// at equal performance levels, so imposing one "correct" cadence number
// would be exactly the false-precision mistake this tool is trying to avoid.
// Only cadence *consistency* (stride-to-stride variability) is scored,
// since an erratic rhythm is a meaningful fatigue/technique signal regardless
// of an individual's baseline style.
// ---------------------------------------------------------------------------

export interface AngleSample {
  angle: number;
  timestampSeconds: number;
}

/** Minimum angle difference from both neighbors to count as a real peak, not landmark jitter. */
const STEP_MIN_PROMINENCE_DEG = 8;
/** No human sprinter exceeds ~8 Hz step frequency; anything faster is a false detection. */
const STEP_REFRACTORY_SECONDS = 0.12;

/**
 * Generic online local-maximum detector: true if the middle of three
 * consecutive (value, timestamp) samples is a real peak — high enough above
 * both neighbors (`minProminence`, rejects landmark jitter) and far enough
 * past the last confirmed peak (`refractorySeconds`, rejects re-detecting
 * the same physical event from noise). Despite the field name `angle`,
 * `AngleSample` is really just "a number at a time" — this same shape and
 * detector serves knee-angle peaks (steps, toe-off) and ankle-height peaks
 * (ground contact) below.
 */
export function isLocalPeak(
  samples: readonly [AngleSample, AngleSample, AngleSample],
  lastPeakTimestampSeconds: number | null,
  minProminence: number,
  refractorySeconds: number
): boolean {
  const [prev, mid, next] = samples;

  const isLocalMax = mid.angle > prev.angle && mid.angle > next.angle;
  if (!isLocalMax) return false;

  const prominence = mid.angle - Math.max(prev.angle, next.angle);
  if (prominence < minProminence) return false;

  if (
    lastPeakTimestampSeconds !== null &&
    mid.timestampSeconds - lastPeakTimestampSeconds < refractorySeconds
  ) {
    return false;
  }

  return true;
}

/**
 * Returns true if the middle of three consecutive knee-drive-angle samples
 * is a local peak — i.e. one leg reaching its maximum drive point, which is
 * one "step" in gait-analysis terminology (a full stride is two steps, one
 * per leg, so this measures step frequency, not stride frequency).
 */
export function isStepPeak(
  samples: readonly [AngleSample, AngleSample, AngleSample],
  lastPeakTimestampSeconds: number | null
): boolean {
  return isLocalPeak(samples, lastPeakTimestampSeconds, STEP_MIN_PROMINENCE_DEG, STEP_REFRACTORY_SECONDS);
}

// ---------------------------------------------------------------------------
// Gait event detection: ground contact and toe-off
//
// "Knee extension at toe-off" and "over-stride distance at ground contact"
// both need to know *which frames* are ground-contact and toe-off instants
// first — there's no force plate here, so both are estimated from the
// vision-only landmark trajectories, the same well-precedented
// simplification used in markerless sprint analysis generally.
//
// Ground contact ~= a local maximum in the ankle's *image* y-coordinate.
// Image y increases downward, so the ankle is closest to the ground (in the
// picture) when y is largest — this is a "how high up the screen" question,
// so it uses the 2D image landmarks on purpose, for the same reason trunk
// lean and lead/trail-leg detection do (see the file-level 2D-vs-3D note):
// the 2D image y-axis is an unambiguous, documented "down" convention,
// where the 3D world-landmark axes are not documented as gravity-aligned.
//
// Toe-off ~= the local maximum of that same leg's knee-extension-angle
// trajectory shortly after its ground contact. Physiologically, the knee
// flexes slightly at midstance and then extends rapidly to near-full
// extension right at push-off, so the peak of that extension curve shortly
// after footstrike is a reasonable, commonly-used proxy for toe-off timing
// without ground-truth data.
// ---------------------------------------------------------------------------

/** Minimum rise in normalized image-y (fraction of frame height) to count as a real footstrike, not jitter. */
const GROUND_CONTACT_MIN_PROMINENCE = 0.01;
/** No sprinter re-contacts the same foot faster than this. */
const GROUND_CONTACT_REFRACTORY_SECONDS = 0.12;

/** Returns true if the middle sample is a ground-contact instant for one ankle's image-y trajectory. */
export function isGroundContactPeak(
  samples: readonly [AngleSample, AngleSample, AngleSample],
  lastContactTimestampSeconds: number | null
): boolean {
  return isLocalPeak(
    samples,
    lastContactTimestampSeconds,
    GROUND_CONTACT_MIN_PROMINENCE,
    GROUND_CONTACT_REFRACTORY_SECONDS
  );
}

/** Minimum degrees of rise to count a knee-angle peak as a real toe-off, not jitter. */
export const TOE_OFF_MIN_PROMINENCE_DEG = 5;
/** No sprinter's same leg toes off again faster than this. */
export const TOE_OFF_REFRACTORY_SECONDS = 0.15;
/** How long after a detected ground contact to keep watching for that leg's toe-off before giving up. */
export const TOE_OFF_SEARCH_WINDOW_SECONDS = 0.35;
/**
 * Upper bound on a plausible single-leg flight time (that leg's toe-off to
 * its own next ground contact) — guards against pairing a ground contact
 * with a stale toe-off from a much earlier, undetected stride (e.g. one
 * where the toe-off search window above expired without a confirmed peak)
 * and reporting a nonsensical multi-stride "flight time" as a result.
 */
export const FLIGHT_TIME_MAX_SECONDS = 0.5;
/** Lower bound, same irregular-processing-cadence reasoning as GROUND_CONTACT_TIME_MIN_SECONDS below. */
export const FLIGHT_TIME_MIN_SECONDS = 0.1;
/**
 * Lower bound on a plausible ground contact time. Even the fastest recorded
 * elite sprinters bottom out around 0.08s at max velocity, and TOE_OFF_
 * REFRACTORY_SECONDS above already prevents two *real* toe-offs closer than
 * 0.15s apart on the same leg — so a stance duration under this floor isn't
 * a fast contact, it's near-certainly two samples from an irregular
 * processing cadence (e.g. slow inference throughput skipping real frames)
 * being mistaken for adjacent-in-time. Rejected rather than clamped, since a
 * clamped-but-wrong value would misleadingly look like real data.
 */
export const GROUND_CONTACT_TIME_MIN_SECONDS = 0.04;
/** Upper bound, mirroring FLIGHT_TIME_MAX_SECONDS's reasoning for the other half of the stride cycle. */
export const GROUND_CONTACT_TIME_MAX_SECONDS = 0.4;

/** Returns true if the middle sample is a toe-off instant for one leg's knee-extension-angle trajectory. */
export function isToeOffPeak(
  samples: readonly [AngleSample, AngleSample, AngleSample],
  lastToeOffTimestampSeconds: number | null
): boolean {
  return isLocalPeak(samples, lastToeOffTimestampSeconds, TOE_OFF_MIN_PROMINENCE_DEG, TOE_OFF_REFRACTORY_SECONDS);
}

/** One SessionAggregates bucket per sprint phase, so stats never blend across phases. */
export type PhaseAggregates = Record<SprintPhase, SessionAggregates>;

export function createPhaseAggregates(): PhaseAggregates {
  return {
    acceleration: createSessionAggregates(),
    transition: createSessionAggregates(),
    maxVelocity: createSessionAggregates(),
  };
}

/** Folds one frame's computed angles into its phase's running aggregates. */
export function recordFrameMetrics(aggregates: PhaseAggregates, frame: FrameMetrics): void {
  const bucket = aggregates[frame.phase];
  if (frame.trunkLeanAngle !== null) addSampleInPlace(bucket.trunkLean, frame.trunkLeanAngle);
  if (frame.kneeDriveAngle !== null) {
    addSampleInPlace(bucket.kneeDrive, frame.kneeDriveAngle);
    if (frame.kneeDriveSide === 'left') addSampleInPlace(bucket.kneeDriveLeft, frame.kneeDriveAngle);
    else if (frame.kneeDriveSide === 'right') addSampleInPlace(bucket.kneeDriveRight, frame.kneeDriveAngle);
  }
  if (frame.hipExtensionAngle !== null) addSampleInPlace(bucket.hipExtension, frame.hipExtensionAngle);
  if (frame.leftArmSwingAngle !== null) addSampleInPlace(bucket.leftArmSwing, frame.leftArmSwingAngle);
  if (frame.rightArmSwingAngle !== null) addSampleInPlace(bucket.rightArmSwing, frame.rightArmSwingAngle);
}

// ---------------------------------------------------------------------------
// Continuous scoring (0-100)
//
// classifyByLowerBound/classifyByTargetRange above are deliberately coarse
// (3 buckets) for a glanceable live overlay. A phase verdict deserves more
// resolution, so these map the *same* threshold constants onto a continuous
// scale: 100 at the optimal boundary, 50 at the caution boundary, 0 one
// caution-to-optimal span beyond that — clamped to [0, 100].
// ---------------------------------------------------------------------------

function clampScore(score: number): number {
  return Math.min(100, Math.max(0, score));
}

function scoreLowerBoundMetric(angle: number, cautionMin: number, optimalMin: number): number {
  const span = optimalMin - cautionMin;
  if (span <= 0) return angle >= optimalMin ? 100 : 0;
  return clampScore(50 + ((angle - cautionMin) / span) * 50);
}

/** Mirror of scoreLowerBoundMetric for "smaller is better beyond this point" metrics. */
function scoreUpperBoundMetric(angle: number, optimalMax: number, cautionMax: number): number {
  const span = cautionMax - optimalMax;
  if (span <= 0) return angle <= optimalMax ? 100 : 0;
  return clampScore(50 + ((cautionMax - angle) / span) * 50);
}

function scoreTargetRangeMetric(
  angle: number,
  cautionMin: number,
  optimalMin: number,
  optimalMax: number,
  cautionMax: number
): number {
  if (angle >= optimalMin && angle <= optimalMax) return 100;
  if (angle < optimalMin) return scoreLowerBoundMetric(angle, cautionMin, optimalMin);
  return scoreUpperBoundMetric(angle, optimalMax, cautionMax);
}

function statusFromScore(score: number): MetricStatus {
  if (score >= 80) return 'optimal';
  if (score >= 55) return 'caution';
  return 'suboptimal';
}

export interface MetricScore {
  label: string;
  averageAngle: number;
  stdDevAngle: number | null;
  sampleCount: number;
  score: number;
  status: MetricStatus;
}

function buildMetricScore(
  label: string,
  aggregate: MetricAggregate,
  scoreAngle: (angle: number) => number
): MetricScore | null {
  const mean = aggregateMean(aggregate);
  if (mean === null) return null;
  const score = scoreAngle(mean);
  return {
    label,
    averageAngle: mean,
    stdDevAngle: aggregateStdDev(aggregate),
    sampleCount: aggregate.count,
    score,
    status: statusFromScore(score),
  };
}

// ---------------------------------------------------------------------------
// Phase summaries: per-phase score + coaching recommendations
// ---------------------------------------------------------------------------

/** Below this many samples, an average is too noisy to score with confidence. */
export const MIN_SAMPLES_FOR_SUMMARY = 15;
/** Step frequency samples are one-per-step (not one-per-frame), so a much smaller floor applies. */
export const MIN_STEPS_FOR_CADENCE = 4;
/** Ground-contact/toe-off events are one-per-stride, same reasoning as MIN_STEPS_FOR_CADENCE. */
export const MIN_GAIT_EVENTS_FOR_SUMMARY = 3;

const HIGH_VARIABILITY_STDDEV_DEG = 15;
/** Below this stride-to-stride stddev, consistency itself is worth calling out as a strength. */
const LOW_VARIABILITY_STDDEV_DEG = 8;
const ARM_ASYMMETRY_WARNING_DEG = 15;
/** Below this left/right gap, symmetry itself is worth calling out as a strength. */
const ARM_SYMMETRY_STRENGTH_DEG = 6;
/** Wider tolerance than arm symmetry: knee drive has a larger natural dynamic range across phases. */
const LEG_ASYMMETRY_WARNING_DEG = 20;
const LEG_SYMMETRY_STRENGTH_DEG = 8;
/** Coefficient of variation (stdDev/mean) above which cadence is "inconsistent," not just "your style." */
const STEP_FREQUENCY_CV_WARNING = 0.15;

export interface Recommendation {
  id: string;
  severity: Extract<MetricStatus, 'caution' | 'suboptimal'>;
  message: string;
}

/** A specific, numbers-backed positive callout — the mirror image of a Recommendation. */
export interface Strength {
  id: string;
  message: string;
}

/**
 * Deliberately has no `score`/`status` — see the step-cadence-detection
 * comment above for why an absolute "optimal" cadence isn't imposed.
 */
export interface StepFrequencyStats {
  averageHz: number;
  stdDevHz: number | null;
  sampleCount: number;
}

/**
 * Deliberately has no `score`/`status`, same reasoning as StepFrequencyStats:
 * flight (swing) time shrinks with speed alongside ground contact time, but
 * duty factor (the ground-contact/flight split) is what actually shifts with
 * technique — flight time alone isn't independently "better" longer or
 * shorter, so it's reported for context rather than graded.
 */
export interface FlightTimeStats {
  averageSeconds: number;
  stdDevSeconds: number | null;
  sampleCount: number;
}

export interface SessionSummary {
  phase: SprintPhase;
  sampleCount: number;
  isReliable: boolean;
  overallScore: number | null;
  overallStatus: MetricStatus | null;
  trunkLean: MetricScore | null;
  kneeDrive: MetricScore | null;
  hipExtension: MetricScore | null;
  leftArmSwing: MetricScore | null;
  rightArmSwing: MetricScore | null;
  armSymmetryDelta: number | null;
  legSymmetryDelta: number | null;
  stepFrequency: StepFrequencyStats | null;
  /**
   * Event-triggered (one sample per detected stride, not per frame) —
   * excluded from overallScore's blend since gait events accumulate far
   * more slowly than continuous frame data, so a phase could clear its
   * frame-based isReliable bar while still having only one or two of
   * these; averaging that sparse a signal into the headline score would
   * make it needlessly volatile. Each still has its own MIN_GAIT_EVENTS_FOR_SUMMARY
   * gate before being shown or recommended on.
   */
  kneeExtensionAtToeOff: MetricScore | null;
  overstride: MetricScore | null;
  /** Same event-triggered exclusion from overallScore as kneeExtensionAtToeOff/overstride above. */
  groundContactTime: MetricScore | null;
  /** Informational only — see FlightTimeStats doc. */
  flightTime: FlightTimeStats | null;
  recommendations: Recommendation[];
  strengths: Strength[];
}

/**
 * Reduces one phase's running aggregates into a 0-100 score and a list of
 * specific, numbers-backed coaching recommendations for whichever metrics
 * are pulling that phase's score down. Pure and O(1) — safe to call every
 * analyzed frame since it only ever reads already-reduced aggregates.
 */
function computePhaseSummary(phase: SprintPhase, aggregates: SessionAggregates): SessionSummary {
  const thresholds = PHASE_THRESHOLDS[phase];
  const phaseLabel = PHASE_LABELS[phase].toLowerCase();

  const trunkLean = buildMetricScore('Trunk Lean', aggregates.trunkLean, (angle) =>
    scoreTargetRangeMetric(
      angle,
      thresholds.trunkLean.cautionMin,
      thresholds.trunkLean.optimalMin,
      thresholds.trunkLean.optimalMax,
      thresholds.trunkLean.cautionMax
    )
  );
  const kneeDrive = buildMetricScore('Knee Drive', aggregates.kneeDrive, (angle) =>
    scoreLowerBoundMetric(angle, thresholds.kneeDrive.cautionMin, thresholds.kneeDrive.optimalMin)
  );
  const hipExtension = buildMetricScore('Hip Extension', aggregates.hipExtension, (angle) =>
    scoreLowerBoundMetric(angle, thresholds.hipExtension.cautionMin, thresholds.hipExtension.optimalMin)
  );
  const leftArmSwing = buildMetricScore('Left Arm Swing', aggregates.leftArmSwing, (angle) =>
    scoreTargetRangeMetric(
      angle,
      thresholds.armSwing.cautionMin,
      thresholds.armSwing.optimalMin,
      thresholds.armSwing.optimalMax,
      thresholds.armSwing.cautionMax
    )
  );
  const rightArmSwing = buildMetricScore('Right Arm Swing', aggregates.rightArmSwing, (angle) =>
    scoreTargetRangeMetric(
      angle,
      thresholds.armSwing.cautionMin,
      thresholds.armSwing.optimalMin,
      thresholds.armSwing.optimalMax,
      thresholds.armSwing.cautionMax
    )
  );

  // Event-triggered, not per-frame — see the SessionSummary field doc for
  // why these are deliberately excluded from the `scores` blend below.
  const kneeExtensionAtToeOffRaw = buildMetricScore(
    'Knee Extension at Toe-Off',
    aggregates.kneeExtensionAtToeOff,
    (angle) =>
      scoreLowerBoundMetric(
        angle,
        thresholds.kneeExtensionAtToeOff.cautionMin,
        thresholds.kneeExtensionAtToeOff.optimalMin
      )
  );
  const kneeExtensionAtToeOff =
    kneeExtensionAtToeOffRaw && kneeExtensionAtToeOffRaw.sampleCount >= MIN_GAIT_EVENTS_FOR_SUMMARY
      ? kneeExtensionAtToeOffRaw
      : null;

  const overstrideRaw = buildMetricScore('Over-stride', aggregates.overstride, (distance) =>
    scoreUpperBoundMetric(distance, OVERSTRIDE_CAUTION_M, OVERSTRIDE_SUBOPTIMAL_M)
  );
  const overstride =
    overstrideRaw && overstrideRaw.sampleCount >= MIN_GAIT_EVENTS_FOR_SUMMARY ? overstrideRaw : null;

  const groundContactTimeRaw = buildMetricScore(
    'Ground Contact Time',
    aggregates.groundContactTime,
    (seconds) =>
      scoreUpperBoundMetric(seconds, thresholds.groundContactTime.optimalMax, thresholds.groundContactTime.cautionMax)
  );
  const groundContactTime =
    groundContactTimeRaw && groundContactTimeRaw.sampleCount >= MIN_GAIT_EVENTS_FOR_SUMMARY
      ? groundContactTimeRaw
      : null;

  // Informational only (see FlightTimeStats doc) — built directly from the
  // aggregate rather than buildMetricScore/scoreAngle, same as stepFrequency.
  const flightTimeMean = aggregateMean(aggregates.flightTime);
  const flightTime: FlightTimeStats | null =
    flightTimeMean !== null && aggregates.flightTime.count >= MIN_GAIT_EVENTS_FOR_SUMMARY
      ? {
          averageSeconds: flightTimeMean,
          stdDevSeconds: aggregateStdDev(aggregates.flightTime),
          sampleCount: aggregates.flightTime.count,
        }
      : null;

  const scores = [trunkLean, kneeDrive, hipExtension, leftArmSwing, rightArmSwing].filter(
    (s): s is MetricScore => s !== null
  );

  const sampleCount = Math.max(
    trunkLean?.sampleCount ?? 0,
    kneeDrive?.sampleCount ?? 0,
    hipExtension?.sampleCount ?? 0,
    leftArmSwing?.sampleCount ?? 0,
    rightArmSwing?.sampleCount ?? 0
  );
  const isReliable = sampleCount >= MIN_SAMPLES_FOR_SUMMARY;

  const overallScore =
    isReliable && scores.length > 0
      ? scores.reduce((sum, s) => sum + s.score, 0) / scores.length
      : null;
  const overallStatus = overallScore !== null ? statusFromScore(overallScore) : null;

  const armSymmetryDelta =
    leftArmSwing && rightArmSwing ? Math.abs(leftArmSwing.averageAngle - rightArmSwing.averageAngle) : null;

  const kneeDriveLeftMean = aggregateMean(aggregates.kneeDriveLeft);
  const kneeDriveRightMean = aggregateMean(aggregates.kneeDriveRight);
  const legSymmetryDelta =
    kneeDriveLeftMean !== null && kneeDriveRightMean !== null
      ? Math.abs(kneeDriveLeftMean - kneeDriveRightMean)
      : null;

  const stepFrequencyMean = aggregateMean(aggregates.stepFrequency);
  const stepFrequency: StepFrequencyStats | null =
    stepFrequencyMean !== null
      ? {
          averageHz: stepFrequencyMean,
          stdDevHz: aggregateStdDev(aggregates.stepFrequency),
          sampleCount: aggregates.stepFrequency.count,
        }
      : null;

  const recommendations: Recommendation[] = [];

  if (isReliable) {
    if (trunkLean && trunkLean.status !== 'optimal') {
      const tooUpright = trunkLean.averageAngle < thresholds.trunkLean.optimalMin;
      const message =
        phase === 'maxVelocity'
          ? `Trunk lean during max velocity averaged ${trunkLean.averageAngle.toFixed(0)}°, more forward lean than elite top-speed posture (roughly 0-10°). Focus on standing tall through the torso — staying flexed forward here can cost top-end speed.`
          : tooUpright
            ? `Trunk lean during ${phaseLabel} averaged only ${trunkLean.averageAngle.toFixed(0)}°, less forward lean than typical elite ${phaseLabel} posture (roughly ${thresholds.trunkLean.optimalMin}-${thresholds.trunkLean.optimalMax}°). Standing up too early limits how much horizontal force you can apply.`
            : `Trunk lean during ${phaseLabel} averaged ${trunkLean.averageAngle.toFixed(0)}°, more than typical elite ${phaseLabel} posture (roughly ${thresholds.trunkLean.optimalMin}-${thresholds.trunkLean.optimalMax}°). Overreaching forward can disrupt ground contact and balance.`;
      recommendations.push({ id: 'trunk-lean', severity: trunkLean.status, message });
    }

    if (kneeDrive && kneeDrive.status !== 'optimal') {
      const context =
        phase === 'maxVelocity'
          ? 'Front-side knee drive is the hallmark of elite top-speed mechanics — a "knee to the sky" cue can help.'
          : 'Knee lift naturally builds through the drive phase as you rise toward top speed — keep driving forward and up.';
      recommendations.push({
        id: 'knee-drive',
        severity: kneeDrive.status,
        message: `Lead-leg knee drive during ${phaseLabel} averaged ${kneeDrive.averageAngle.toFixed(0)}°, below the ${thresholds.kneeDrive.optimalMin}° target for this phase. ${context}`,
      });
    }

    if (hipExtension && hipExtension.status !== 'optimal') {
      const context =
        phase === 'acceleration'
          ? 'Acceleration relies on completing full hip extension to apply horizontal force — push the ground away completely before recovering the leg.'
          : 'Aim for a fuller, more powerful push off the ground each stride.';
      recommendations.push({
        id: 'hip-extension',
        severity: hipExtension.status,
        message: `Trail-leg hip extension during ${phaseLabel} averaged ${hipExtension.averageAngle.toFixed(0)}°, short of the ${thresholds.hipExtension.optimalMin}° target for this phase. ${context}`,
      });
    }

    const armSides: Array<['Left' | 'Right', MetricScore | null]> = [
      ['Left', leftArmSwing],
      ['Right', rightArmSwing],
    ];
    for (const [side, armScore] of armSides) {
      if (armScore && armScore.status !== 'optimal') {
        const tooBent = armScore.averageAngle < thresholds.armSwing.optimalMin;
        const fix = tooBent
          ? 'relax the elbow slightly to open the swing'
          : 'keep a tighter elbow bend to maintain swing speed';
        recommendations.push({
          id: `arm-swing-${side.toLowerCase()}`,
          severity: armScore.status,
          message: `${side} arm swing during ${phaseLabel} averaged ${armScore.averageAngle.toFixed(0)}° at the elbow — ${
            tooBent ? 'too bent' : 'too extended'
          } relative to the ${thresholds.armSwing.optimalMin}–${thresholds.armSwing.optimalMax}° target for this phase. Try to ${fix}.`,
        });
      }
    }

    if (armSymmetryDelta !== null && armSymmetryDelta > ARM_ASYMMETRY_WARNING_DEG) {
      recommendations.push({
        id: 'arm-symmetry',
        severity: 'caution',
        message: `Left and right arm swing differ by ${armSymmetryDelta.toFixed(0)}° on average during ${phaseLabel}, which can point to rotational compensation or a strength imbalance. Unilateral upper-body work may help even this out.`,
      });
    }

    if (
      legSymmetryDelta !== null &&
      legSymmetryDelta > LEG_ASYMMETRY_WARNING_DEG &&
      kneeDriveLeftMean !== null &&
      kneeDriveRightMean !== null
    ) {
      const weakerSide = kneeDriveLeftMean < kneeDriveRightMean ? 'left' : 'right';
      recommendations.push({
        id: 'leg-symmetry',
        severity: 'caution',
        message: `Knee drive differs by ${legSymmetryDelta.toFixed(0)}° between legs during ${phaseLabel} (${weakerSide} side lower on average) — a persistent gap like this can reflect a strength imbalance or a compensation pattern from a previous injury, and is worth a closer look.`,
      });
    }

    if (kneeDrive && kneeDrive.stdDevAngle !== null && kneeDrive.stdDevAngle > HIGH_VARIABILITY_STDDEV_DEG) {
      recommendations.push({
        id: 'knee-drive-consistency',
        severity: 'caution',
        message: `Knee drive varied by ±${kneeDrive.stdDevAngle.toFixed(0)}° stride to stride during ${phaseLabel} — high variability can signal fatigue or inconsistent technique.`,
      });
    }

    if (
      hipExtension &&
      hipExtension.stdDevAngle !== null &&
      hipExtension.stdDevAngle > HIGH_VARIABILITY_STDDEV_DEG
    ) {
      recommendations.push({
        id: 'hip-extension-consistency',
        severity: 'caution',
        message: `Hip extension varied by ±${hipExtension.stdDevAngle.toFixed(0)}° stride to stride during ${phaseLabel} — aim for a more repeatable push-off each stride.`,
      });
    }

    if (
      stepFrequency &&
      stepFrequency.sampleCount >= MIN_STEPS_FOR_CADENCE &&
      stepFrequency.stdDevHz !== null &&
      stepFrequency.stdDevHz / stepFrequency.averageHz > STEP_FREQUENCY_CV_WARNING
    ) {
      recommendations.push({
        id: 'cadence-consistency',
        severity: 'caution',
        message: `Step frequency varied noticeably during ${phaseLabel} (averaged ${stepFrequency.averageHz.toFixed(1)} Hz, ±${stepFrequency.stdDevHz.toFixed(1)} Hz) — an unsteady rhythm can signal fatigue or a breakdown in timing. There's no single "correct" cadence (elite sprinters are legitimately frequency-reliant or length-reliant), but holding it steady matters.`,
      });
    }

    if (kneeExtensionAtToeOff && kneeExtensionAtToeOff.status !== 'optimal') {
      const context =
        phase === 'maxVelocity'
          ? 'At max velocity this is less critical than during acceleration — elite sprinters intentionally favor a quick, elastic ground contact over reaching full extension.'
          : 'Push through the full range at push-off rather than leaving the ground early.';
      recommendations.push({
        id: 'knee-extension-toe-off',
        severity: kneeExtensionAtToeOff.status,
        message: `Knee extension at toe-off during ${phaseLabel} averaged ${kneeExtensionAtToeOff.averageAngle.toFixed(0)}°, below the ${thresholds.kneeExtensionAtToeOff.optimalMin}° target for this phase (based on ${kneeExtensionAtToeOff.sampleCount} detected toe-offs). ${context}`,
      });
    }

    if (overstride && overstride.status !== 'optimal') {
      recommendations.push({
        id: 'overstride',
        severity: overstride.status,
        message: `Over-striding during ${phaseLabel}: the foot landed ${overstride.averageAngle.toFixed(2)}m ahead of the hip center on average (based on ${overstride.sampleCount} detected ground contacts) — landing well ahead of the center of mass creates a braking force at footstrike. Aim to land with the foot closer to underneath the hips.`,
      });
    }

    if (groundContactTime && groundContactTime.status !== 'optimal') {
      const context =
        phase === 'acceleration'
          ? 'Some extra ground time here is expected while driving out of the start, but too much suggests you\'re pushing rather than driving explosively.'
          : 'A quick, elastic ground contact — minimizing time on the ground — is a hallmark of efficient sprinting at this phase.';
      recommendations.push({
        id: 'ground-contact-time',
        severity: groundContactTime.status,
        message: `Ground contact time during ${phaseLabel} averaged ${(groundContactTime.averageAngle * 1000).toFixed(0)}ms, longer than the ${(thresholds.groundContactTime.optimalMax * 1000).toFixed(0)}ms target for this phase (based on ${groundContactTime.sampleCount} detected strides). ${context}`,
      });
    }

    recommendations.sort((a, b) => {
      if (a.severity === b.severity) return 0;
      return a.severity === 'suboptimal' ? -1 : 1;
    });
  }

  const strengths: Strength[] = [];

  if (isReliable) {
    if (trunkLean && trunkLean.status === 'optimal') {
      strengths.push({
        id: 'trunk-lean',
        message:
          phase === 'maxVelocity'
            ? `Trunk lean averaged ${trunkLean.averageAngle.toFixed(0)}° — tall, upright posture, exactly what elite top-speed running looks like.`
            : `Trunk lean averaged ${trunkLean.averageAngle.toFixed(0)}°, right in the elite ${phaseLabel} range (${thresholds.trunkLean.optimalMin}-${thresholds.trunkLean.optimalMax}°) — strong drive posture.`,
      });
    }

    if (kneeDrive && kneeDrive.status === 'optimal') {
      strengths.push({
        id: 'knee-drive',
        message: `Lead-leg knee drive averaged ${kneeDrive.averageAngle.toFixed(0)}°, above the ${thresholds.kneeDrive.optimalMin}° elite target for ${phaseLabel} — strong front-side mechanics.`,
      });
    }

    if (hipExtension && hipExtension.status === 'optimal') {
      strengths.push({
        id: 'hip-extension',
        message: `Trail-leg hip extension averaged ${hipExtension.averageAngle.toFixed(0)}°, meeting the ${thresholds.hipExtension.optimalMin}° target for ${phaseLabel} — you're completing the push effectively.`,
      });
    }

    const optimalArmSides: Array<['Left' | 'Right', MetricScore | null]> = [
      ['Left', leftArmSwing],
      ['Right', rightArmSwing],
    ];
    for (const [side, armScore] of optimalArmSides) {
      if (armScore && armScore.status === 'optimal') {
        strengths.push({
          id: `arm-swing-${side.toLowerCase()}`,
          message: `${side} arm swing averaged ${armScore.averageAngle.toFixed(0)}° at the elbow — efficient, compact mechanics for ${phaseLabel}.`,
        });
      }
    }

    if (armSymmetryDelta !== null && armSymmetryDelta < ARM_SYMMETRY_STRENGTH_DEG) {
      strengths.push({
        id: 'arm-symmetry',
        message: `Left and right arm swing were nearly identical (±${armSymmetryDelta.toFixed(0)}°) during ${phaseLabel} — well-balanced upper-body mechanics.`,
      });
    }

    if (legSymmetryDelta !== null && legSymmetryDelta < LEG_SYMMETRY_STRENGTH_DEG) {
      strengths.push({
        id: 'leg-symmetry',
        message: `Knee drive was nearly even between legs (±${legSymmetryDelta.toFixed(0)}°) during ${phaseLabel} — no signs of a left/right compensation pattern.`,
      });
    }

    if (kneeDrive && kneeDrive.stdDevAngle !== null && kneeDrive.stdDevAngle < LOW_VARIABILITY_STDDEV_DEG) {
      strengths.push({
        id: 'knee-drive-consistency',
        message: `Knee drive was remarkably consistent stride to stride (±${kneeDrive.stdDevAngle.toFixed(0)}°) during ${phaseLabel} — repeatable technique.`,
      });
    }

    if (
      hipExtension &&
      hipExtension.stdDevAngle !== null &&
      hipExtension.stdDevAngle < LOW_VARIABILITY_STDDEV_DEG
    ) {
      strengths.push({
        id: 'hip-extension-consistency',
        message: `Hip extension was remarkably consistent stride to stride (±${hipExtension.stdDevAngle.toFixed(0)}°) during ${phaseLabel} — a repeatable push-off.`,
      });
    }

    if (kneeExtensionAtToeOff && kneeExtensionAtToeOff.status === 'optimal') {
      strengths.push({
        id: 'knee-extension-toe-off',
        message: `Knee extension at toe-off during ${phaseLabel} averaged ${kneeExtensionAtToeOff.averageAngle.toFixed(0)}° across ${kneeExtensionAtToeOff.sampleCount} detected toe-offs — a strong, complete push-off.`,
      });
    }

    if (overstride && overstride.status === 'optimal') {
      strengths.push({
        id: 'overstride',
        message: `Ground contacts during ${phaseLabel} landed close to underneath the hips (${overstride.averageAngle.toFixed(2)}m average gap across ${overstride.sampleCount} contacts) — minimal braking force at footstrike.`,
      });
    }

    if (groundContactTime && groundContactTime.status === 'optimal') {
      strengths.push({
        id: 'ground-contact-time',
        message: `Ground contact time during ${phaseLabel} averaged ${(groundContactTime.averageAngle * 1000).toFixed(0)}ms across ${groundContactTime.sampleCount} detected strides — quick, elastic contact with the ground.`,
      });
    }
  }

  return {
    phase,
    sampleCount,
    isReliable,
    overallScore,
    overallStatus,
    trunkLean,
    kneeDrive,
    hipExtension,
    leftArmSwing,
    rightArmSwing,
    armSymmetryDelta,
    legSymmetryDelta,
    stepFrequency,
    kneeExtensionAtToeOff,
    overstride,
    groundContactTime,
    flightTime,
    recommendations,
    strengths,
  };
}

/**
 * Computes one SessionSummary per phase that has at least one analyzed
 * frame. A phase the clip never showed (e.g. a max-velocity-only fly-in
 * clip never touching acceleration) simply has no entry — its score is
 * never blended into anything else's, which is the whole point of scoring
 * per phase instead of averaging across the entire clip.
 */
export function computePhaseSummaries(aggregates: PhaseAggregates): Partial<Record<SprintPhase, SessionSummary>> {
  const result: Partial<Record<SprintPhase, SessionSummary>> = {};
  for (const phase of SPRINT_PHASES) {
    const summary = computePhaseSummary(phase, aggregates[phase]);
    // sampleCount only reflects the five continuous per-frame metrics (see
    // its computation above) — a phase can still have genuine data worth
    // showing from gait events alone (rare in practice, since detecting
    // even one full gait event requires far more frames than
    // MIN_SAMPLES_FOR_SUMMARY, but not impossible), so the inclusion check
    // considers those too rather than silently dropping the phase.
    const hasAnyData =
      summary.sampleCount > 0 ||
      summary.kneeExtensionAtToeOff !== null ||
      summary.overstride !== null ||
      summary.groundContactTime !== null ||
      summary.flightTime !== null;
    if (hasAnyData) result[phase] = summary;
  }
  return result;
}

/**
 * Synthesizes a short, coach-style paragraph across every reliably-scored
 * phase in the clip — the one narrative thread tying together what would
 * otherwise be several isolated phase report cards. Returns null until at
 * least one phase has enough samples to say anything with confidence.
 */
export function generateSessionNarrative(
  summaries: Partial<Record<SprintPhase, SessionSummary>>
): string | null {
  const reliable = SPRINT_PHASES.map((phase) => summaries[phase]).filter(
    (s): s is SessionSummary => !!s && s.isReliable
  );

  if (reliable.length === 0) return null;

  if (reliable.length === 1) {
    const summary = reliable[0];
    const label = PHASE_LABELS[summary.phase].toLowerCase();
    const scoreText = summary.overallScore !== null ? `${Math.round(summary.overallScore)}/100` : 'not yet scored';
    const topIssue = summary.recommendations[0];
    const topStrength = summary.strengths[0];

    if (topIssue) {
      return `Your ${label} phase scored ${scoreText}. Biggest opportunity: ${topIssue.message}`;
    }
    if (topStrength) {
      return `Your ${label} phase scored ${scoreText}, with no major issues detected. Standout: ${topStrength.message}`;
    }
    return `Your ${label} phase scored ${scoreText}, with no major issues detected.`;
  }

  const sorted = [...reliable].sort((a, b) => (b.overallScore ?? 0) - (a.overallScore ?? 0));
  const strongest = sorted[0];
  const weakest = sorted[sorted.length - 1];
  const strongestLabel = PHASE_LABELS[strongest.phase].toLowerCase();
  const weakestLabel = PHASE_LABELS[weakest.phase].toLowerCase();

  let narrative = `Your ${strongestLabel} phase is your strongest (${
    strongest.overallScore !== null ? Math.round(strongest.overallScore) : '—'
  }/100).`;

  if (weakest.phase !== strongest.phase) {
    narrative += ` Your ${weakestLabel} phase trails at ${
      weakest.overallScore !== null ? Math.round(weakest.overallScore) : '—'
    }/100.`;
    const topIssue = weakest.recommendations[0];
    narrative += topIssue
      ? ` Focus there first: ${topIssue.message}`
      : ' No major issues there either — the gap is likely just normal stride-to-stride variation.';
  }

  return narrative;
}

// ---------------------------------------------------------------------------
// CSV export
// ---------------------------------------------------------------------------

const CSV_HEADERS = [
  'frame_index',
  'timestamp_seconds',
  'sprint_phase',
  'trunk_lean_angle_deg',
  'trunk_lean_status',
  'knee_drive_angle_deg',
  'knee_drive_side',
  'knee_drive_status',
  'hip_extension_angle_deg',
  'hip_extension_side',
  'hip_extension_status',
  'left_arm_swing_angle_deg',
  'left_arm_swing_status',
  'right_arm_swing_angle_deg',
  'right_arm_swing_status',
] as const;

function csvCell(value: string | number | null): string {
  if (value === null) return '';
  return typeof value === 'number' ? value.toFixed(2) : value;
}

function generateSummaryCSVBlock(summary: SessionSummary): string {
  const phaseLabel = PHASE_LABELS[summary.phase];
  const lines = [
    '',
    `# ${phaseLabel} Phase Summary`,
    'metric,average_angle_deg,std_dev_deg,sample_count,score,status',
  ];

  const metricRows: Array<[string, MetricScore | null]> = [
    ['trunk_lean', summary.trunkLean],
    ['knee_drive', summary.kneeDrive],
    ['hip_extension', summary.hipExtension],
    ['left_arm_swing', summary.leftArmSwing],
    ['right_arm_swing', summary.rightArmSwing],
    ['knee_extension_at_toe_off', summary.kneeExtensionAtToeOff],
    ['overstride_m', summary.overstride],
    ['ground_contact_time_s', summary.groundContactTime],
  ];

  for (const [key, metric] of metricRows) {
    if (!metric) {
      lines.push([key, '', '', '0', '', ''].join(','));
      continue;
    }
    lines.push(
      [
        key,
        metric.averageAngle.toFixed(2),
        metric.stdDevAngle !== null ? metric.stdDevAngle.toFixed(2) : '',
        String(metric.sampleCount),
        metric.score.toFixed(1),
        metric.status,
      ].join(',')
    );
  }

  lines.push(
    [
      'overall',
      '',
      '',
      String(summary.sampleCount),
      summary.overallScore !== null ? summary.overallScore.toFixed(1) : '',
      summary.overallStatus ?? '',
    ].join(',')
  );

  if (summary.stepFrequency) {
    lines.push(
      '',
      'step_frequency_hz,std_dev_hz,sample_count',
      [
        summary.stepFrequency.averageHz.toFixed(2),
        summary.stepFrequency.stdDevHz !== null ? summary.stepFrequency.stdDevHz.toFixed(2) : '',
        String(summary.stepFrequency.sampleCount),
      ].join(',')
    );
  }

  if (summary.flightTime) {
    lines.push(
      '',
      'flight_time_s,std_dev_s,sample_count',
      [
        summary.flightTime.averageSeconds.toFixed(3),
        summary.flightTime.stdDevSeconds !== null ? summary.flightTime.stdDevSeconds.toFixed(3) : '',
        String(summary.flightTime.sampleCount),
      ].join(',')
    );
  }

  if (summary.strengths.length > 0) {
    lines.push('', `# ${phaseLabel} Strengths`);
    summary.strengths.forEach((strength) => {
      lines.push(`"${strength.message.replace(/"/g, '""')}"`);
    });
  }

  if (summary.recommendations.length > 0) {
    lines.push('', `# ${phaseLabel} Recommendations`);
    summary.recommendations.forEach((rec) => {
      lines.push(`"${rec.message.replace(/"/g, '""')}"`);
    });
  }

  return lines.join('\n');
}

/**
 * Serializes accumulated frame metrics into an RFC 4180-friendly CSV string.
 * When `phaseSummaries` is provided, one summary block (overall score,
 * per-metric averages, and recommendations) is appended per phase present
 * in the clip, after the frame-by-frame rows.
 */
export function generateCSV(
  frames: FrameMetrics[],
  phaseSummaries?: Partial<Record<SprintPhase, SessionSummary>>
): string {
  const rows = frames.map((frame) =>
    [
      csvCell(frame.frameIndex),
      csvCell(frame.timestampSeconds),
      csvCell(frame.phase),
      csvCell(frame.trunkLeanAngle),
      csvCell(frame.trunkLeanStatus),
      csvCell(frame.kneeDriveAngle),
      csvCell(frame.kneeDriveSide),
      csvCell(frame.kneeDriveStatus),
      csvCell(frame.hipExtensionAngle),
      csvCell(frame.hipExtensionSide),
      csvCell(frame.hipExtensionStatus),
      csvCell(frame.leftArmSwingAngle),
      csvCell(frame.leftArmSwingStatus),
      csvCell(frame.rightArmSwingAngle),
      csvCell(frame.rightArmSwingStatus),
    ].join(',')
  );

  const measurementNote =
    '# trunk_lean_angle_deg is a 2D image-plane angle (against true vertical); knee_drive, hip_extension, and arm_swing angles are computed from 3D real-world landmarks (meters). See README for why.';
  const base = [measurementNote, CSV_HEADERS.join(','), ...rows].join('\n');
  if (!phaseSummaries) return base;

  const narrative = generateSessionNarrative(phaseSummaries);
  const narrativeBlock = narrative
    ? ['', '# Session Summary', `"${narrative.replace(/"/g, '""')}"`].join('\n')
    : '';

  const summaryBlocks = SPRINT_PHASES.filter((phase) => phaseSummaries[phase])
    .map((phase) => generateSummaryCSVBlock(phaseSummaries[phase] as SessionSummary))
    .join('\n');

  return [base, narrativeBlock, summaryBlocks].filter(Boolean).join('\n');
}

/** Triggers a browser download of the given CSV content as a file. */
export function downloadCSV(filename: string, csvContent: string): void {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}
