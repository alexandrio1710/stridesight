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
 * Exponential moving average. Sprint phase shouldn't flicker frame to frame
 * because one frame's landmark estimate is slightly noisy — real phase
 * transitions unfold over roughly a second of running, so fairly heavy
 * smoothing (small alpha) is appropriate.
 */
export function smoothValue(previous: number | null, next: number, alpha = 0.15): number {
  if (previous === null) return next;
  return alpha * next + (1 - alpha) * previous;
}

// ---------------------------------------------------------------------------
// Phase-specific coaching thresholds
// ---------------------------------------------------------------------------

interface LowerBoundThreshold {
  optimalMin: number;
  cautionMin: number;
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
 */
export const PHASE_THRESHOLDS: Record<SprintPhase, PhaseThresholdSet> = {
  acceleration: {
    trunkLean: { optimalMin: 32, optimalMax: 58, cautionMin: 28, cautionMax: 68 },
    kneeDrive: { optimalMin: 72, cautionMin: 58 },
    hipExtension: { optimalMin: 155, cautionMin: 135 },
    armSwing: { optimalMin: 65, optimalMax: 115, cautionMin: 45, cautionMax: 135 },
  },
  transition: {
    trunkLean: { optimalMin: 13, optimalMax: 28, cautionMin: 8, cautionMax: 33 },
    kneeDrive: { optimalMin: 80, cautionMin: 65 },
    hipExtension: { optimalMin: 150, cautionMin: 130 },
    armSwing: { optimalMin: 72, optimalMax: 108, cautionMin: 52, cautionMax: 128 },
  },
  maxVelocity: {
    trunkLean: { optimalMin: 0, optimalMax: 10, cautionMin: 0, cautionMax: 13 },
    kneeDrive: { optimalMin: 90, cautionMin: 75 },
    hipExtension: { optimalMin: 140, cautionMin: 120 },
    armSwing: { optimalMin: 80, optimalMax: 100, cautionMin: 60, cautionMax: 120 },
  },
} as const;

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

/** Degrees of "dead zone" past a phase boundary required before leaving that phase. */
const PHASE_HYSTERESIS_DEG = 3;

/**
 * classifyPhase() alone can flicker if smoothed lean hovers right at a
 * boundary for an extended stretch (e.g. a sprinter easing out of the drive
 * phase very gradually) — even heavy EMA smoothing only damps high-frequency
 * noise, it doesn't stop a signal that's genuinely sitting on a threshold.
 *
 * This wraps classifyPhase with a Schmitt-trigger-style dead zone: once in
 * 'acceleration' or 'maxVelocity', the signal has to cross meaningfully
 * further past the boundary (not just barely) before the phase is allowed to
 * change. This is deliberately asymmetric — entering a phase from
 * 'transition' uses the plain boundary, only *leaving* acceleration/max
 * velocity is damped — because the practical failure mode observed is
 * rapid bouncing between transition and its neighbor, not the reverse.
 */
export function classifyPhaseWithHysteresis(
  smoothedLeanAngle: number,
  previousPhase: SprintPhase | null
): SprintPhase {
  if (previousPhase === 'acceleration') {
    const stayBoundary = PHASE_THRESHOLDS.acceleration.trunkLean.cautionMin - PHASE_HYSTERESIS_DEG;
    if (smoothedLeanAngle >= stayBoundary) return 'acceleration';
  }
  if (previousPhase === 'maxVelocity') {
    const stayBoundary = PHASE_THRESHOLDS.maxVelocity.trunkLean.cautionMax + PHASE_HYSTERESIS_DEG;
    if (smoothedLeanAngle <= stayBoundary) return 'maxVelocity';
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
  };
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
 * Returns true if the middle of three consecutive knee-drive-angle samples
 * is a local peak — i.e. one leg reaching its maximum drive point, which is
 * one "step" in gait-analysis terminology (a full stride is two steps, one
 * per leg, so this measures step frequency, not stride frequency).
 */
export function isStepPeak(
  samples: readonly [AngleSample, AngleSample, AngleSample],
  lastPeakTimestampSeconds: number | null
): boolean {
  const [prev, mid, next] = samples;

  const isLocalMax = mid.angle > prev.angle && mid.angle > next.angle;
  if (!isLocalMax) return false;

  const prominence = mid.angle - Math.max(prev.angle, next.angle);
  if (prominence < STEP_MIN_PROMINENCE_DEG) return false;

  if (
    lastPeakTimestampSeconds !== null &&
    mid.timestampSeconds - lastPeakTimestampSeconds < STEP_REFRACTORY_SECONDS
  ) {
    return false;
  }

  return true;
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
    if (summary.sampleCount > 0) result[phase] = summary;
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
