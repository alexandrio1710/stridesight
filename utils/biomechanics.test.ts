import { describe, expect, it } from 'vitest';
import {
  addSampleInPlace,
  calculateAngle,
  calculateOverstrideDistance,
  calculateTrunkLeanAngle,
  classifyByLowerBound,
  classifyByTargetRange,
  classifyByUpperBound,
  classifyPhase,
  classifyPhaseWithHysteresis,
  computeFrameMetrics,
  computePhaseSummaries,
  createPhaseAggregates,
  determineLegRoles,
  generateCSV,
  isGroundContactPeak,
  isLandmarkReliable,
  isLocalPeak,
  isStepPeak,
  isToeOffPeak,
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
  PHASE_THRESHOLDS,
  POSE_LANDMARK_INDICES,
  SPRINT_PHASES,
  TRUNK_LEAN_PHASE_TIME_CONSTANT_SECONDS,
  type Landmark,
  type PoseLandmarks,
} from './biomechanics';

function landmark(x: number, y: number, z = 0, visibility = 1): Landmark {
  return { x, y, z, visibility };
}

/** A neutral 33-point pose with every landmark present and reliable, safe as a base to override from. */
function basePose(): PoseLandmarks {
  return new Array(33).fill(null).map(() => landmark(0.5, 0.5));
}

describe('calculateAngle', () => {
  it('returns 90 for a right angle', () => {
    const a = { x: 0, y: -1, z: 0 };
    const b = { x: 0, y: 0, z: 0 };
    const c = { x: 1, y: 0, z: 0 };
    expect(calculateAngle(a, b, c)).toBeCloseTo(90, 5);
  });

  it('returns 180 for three collinear points with b in the middle', () => {
    const a = { x: -1, y: 0, z: 0 };
    const b = { x: 0, y: 0, z: 0 };
    const c = { x: 1, y: 0, z: 0 };
    expect(calculateAngle(a, b, c)).toBeCloseTo(180, 5);
  });

  it('returns 0 for a fully folded (doubled-back) angle', () => {
    const a = { x: 1, y: 0, z: 0 };
    const b = { x: 0, y: 0, z: 0 };
    const c = { x: 1, y: 0, z: 0 };
    expect(calculateAngle(a, b, c)).toBeCloseTo(0, 5);
  });

  it('extends correctly into 3D (a 3D right angle off the xy-plane)', () => {
    const a = { x: 0, y: 0, z: -1 };
    const b = { x: 0, y: 0, z: 0 };
    const c = { x: 1, y: 0, z: 0 };
    expect(calculateAngle(a, b, c)).toBeCloseTo(90, 5);
  });

  it('does not throw or return NaN for a zero-length vector', () => {
    const a = { x: 0, y: 0, z: 0 };
    const b = { x: 0, y: 0, z: 0 };
    const c = { x: 1, y: 0, z: 0 };
    expect(calculateAngle(a, b, c)).toBe(0);
  });

  it('clamps floating-point drift past [-1, 1] without producing NaN', () => {
    // Nearly-but-not-quite collinear, the kind of case floating point can push slightly past acos's domain.
    const a = { x: -1, y: 1e-15, z: 0 };
    const b = { x: 0, y: 0, z: 0 };
    const c = { x: 1, y: 0, z: 0 };
    expect(Number.isNaN(calculateAngle(a, b, c))).toBe(false);
  });
});

describe('isLandmarkReliable', () => {
  it('rejects null/undefined', () => {
    expect(isLandmarkReliable(null)).toBe(false);
    expect(isLandmarkReliable(undefined)).toBe(false);
  });

  it('accepts a landmark with no visibility field at all', () => {
    expect(isLandmarkReliable({ x: 0, y: 0, z: 0 })).toBe(true);
  });

  it('rejects visibility below the threshold and accepts at/above it', () => {
    expect(isLandmarkReliable(landmark(0, 0, 0, 0.1))).toBe(false);
    expect(isLandmarkReliable(landmark(0, 0, 0, 0.9))).toBe(true);
  });
});

describe('determineLegRoles', () => {
  it('picks the knee with the smaller image-y (higher on screen) as the lead leg', () => {
    const pose = basePose();
    pose[POSE_LANDMARK_INDICES.LEFT_KNEE] = landmark(0.4, 0.3);
    pose[POSE_LANDMARK_INDICES.RIGHT_KNEE] = landmark(0.6, 0.6);
    expect(determineLegRoles(pose)).toEqual({ leadSide: 'left', trailSide: 'right' });
  });

  it('returns null when a knee is not reliably tracked', () => {
    const pose = basePose();
    pose[POSE_LANDMARK_INDICES.LEFT_KNEE] = landmark(0.4, 0.3, 0, 0.1);
    pose[POSE_LANDMARK_INDICES.RIGHT_KNEE] = landmark(0.6, 0.6);
    expect(determineLegRoles(pose)).toBeNull();
  });
});

describe('calculateTrunkLeanAngle', () => {
  it('reads ~0deg for a perfectly upright torso', () => {
    const pose = basePose();
    pose[POSE_LANDMARK_INDICES.LEFT_SHOULDER] = landmark(0.48, 0.2);
    pose[POSE_LANDMARK_INDICES.RIGHT_SHOULDER] = landmark(0.52, 0.2);
    pose[POSE_LANDMARK_INDICES.LEFT_HIP] = landmark(0.48, 0.5);
    pose[POSE_LANDMARK_INDICES.RIGHT_HIP] = landmark(0.52, 0.5);
    expect(calculateTrunkLeanAngle(pose)).toBeCloseTo(0, 1);
  });

  it('reads a large angle for a strongly forward-leaning torso', () => {
    const pose = basePose();
    // Shoulders shifted well forward (in x) of the hips -> a strong forward lean.
    pose[POSE_LANDMARK_INDICES.LEFT_SHOULDER] = landmark(0.68, 0.2);
    pose[POSE_LANDMARK_INDICES.RIGHT_SHOULDER] = landmark(0.72, 0.2);
    pose[POSE_LANDMARK_INDICES.LEFT_HIP] = landmark(0.48, 0.5);
    pose[POSE_LANDMARK_INDICES.RIGHT_HIP] = landmark(0.52, 0.5);
    const angle = calculateTrunkLeanAngle(pose);
    expect(angle).not.toBeNull();
    expect(angle as number).toBeGreaterThan(30);
  });

  it('returns null when a shoulder or hip is unreliable', () => {
    const pose = basePose();
    pose[POSE_LANDMARK_INDICES.LEFT_SHOULDER] = landmark(0.48, 0.2, 0, 0.1);
    expect(calculateTrunkLeanAngle(pose)).toBeNull();
  });
});

describe('smoothValue vs smoothValueOverTime', () => {
  it('smoothValue ignores elapsed time -- same alpha regardless of dt', () => {
    const a = smoothValue(10, 20, 0.5);
    expect(a).toBeCloseTo(15, 5);
  });

  it('smoothValueOverTime converges faster with a larger dt', () => {
    const shortDt = smoothValueOverTime(10, 20, 0.1, 1.0);
    const longDt = smoothValueOverTime(10, 20, 5.0, 1.0);
    expect(longDt).toBeGreaterThan(shortDt);
    expect(longDt).toBeLessThanOrEqual(20);
  });

  it('smoothValueOverTime returns the previous value for zero or negative dt', () => {
    expect(smoothValueOverTime(10, 20, 0, 1.0)).toBe(10);
    expect(smoothValueOverTime(10, 20, -1, 1.0)).toBe(10);
  });

  it('smoothValueOverTime returns `next` unmodified when previous is null (first sample)', () => {
    expect(smoothValueOverTime(null, 20, 0, 1.0)).toBe(20);
  });
});

describe('phase classification', () => {
  it('classifies the three raw bands correctly', () => {
    expect(classifyPhase(50)).toBe('acceleration');
    expect(classifyPhase(20)).toBe('transition');
    expect(classifyPhase(5)).toBe('maxVelocity');
  });

  it('hysteresis has a genuine dead zone in both directions at the maxVelocity boundary', () => {
    // From maxVelocity, must clear cautionMax + hysteresis to leave -- not just cautionMax.
    const maxVelCautionMax = PHASE_THRESHOLDS.maxVelocity.trunkLean.cautionMax;
    expect(classifyPhaseWithHysteresis(maxVelCautionMax + 1, 'maxVelocity')).toBe('maxVelocity');

    // From transition, must drop *below* cautionMax - hysteresis to enter maxVelocity -- not just cautionMax.
    expect(classifyPhaseWithHysteresis(maxVelCautionMax - 1, 'transition')).toBe('transition');
  });

  it('hysteresis has a genuine dead zone in both directions at the acceleration boundary', () => {
    const accelCautionMin = PHASE_THRESHOLDS.acceleration.trunkLean.cautionMin;
    expect(classifyPhaseWithHysteresis(accelCautionMin - 1, 'acceleration')).toBe('acceleration');
    expect(classifyPhaseWithHysteresis(accelCautionMin + 1, 'transition')).toBe('transition');
  });

  it('does not get stuck: a large enough swing still crosses every boundary', () => {
    expect(classifyPhaseWithHysteresis(60, 'maxVelocity')).toBe('acceleration');
    expect(classifyPhaseWithHysteresis(2, 'acceleration')).toBe('maxVelocity');
  });

  it('is robust to occlusion-style burst noise on a genuinely single-phase signal', () => {
    // A runner steady at ~9deg (max velocity) the whole time, with occasional
    // short 18-24deg bursts (simulated landmark occlusion), sampled at a
    // consistent 30fps. None of it should ever leave maxVelocity.
    let seed = 7;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    let ema: number | null = null;
    let phase: 'acceleration' | 'transition' | 'maxVelocity' | null = null;
    let burstFramesLeft = 0;
    let misclassified = 0;
    const dt = 1 / 30;
    for (let i = 0; i < 900; i++) {
      if (burstFramesLeft === 0 && rand() < 0.03) burstFramesLeft = 2 + Math.floor(rand() * 3);
      const raw = burstFramesLeft > 0 ? 18 + rand() * 6 : 9 + (rand() - 0.5) * 2;
      if (burstFramesLeft > 0) burstFramesLeft--;
      ema = smoothValueOverTime(ema, raw, dt, TRUNK_LEAN_PHASE_TIME_CONSTANT_SECONDS);
      phase = classifyPhaseWithHysteresis(ema, phase);
      if (phase !== 'maxVelocity') misclassified++;
    }
    // Zero is the ideal, but allow a very small margin for the seeded PRNG's variance.
    expect(misclassified).toBeLessThan(5);
  });
});

describe('classifyByLowerBound / classifyByUpperBound / classifyByTargetRange', () => {
  it('classifyByLowerBound: higher is better', () => {
    expect(classifyByLowerBound(100, 90, 70)).toBe('optimal');
    expect(classifyByLowerBound(80, 90, 70)).toBe('caution');
    expect(classifyByLowerBound(60, 90, 70)).toBe('suboptimal');
  });

  it('classifyByUpperBound: lower is better (mirror of classifyByLowerBound)', () => {
    expect(classifyByUpperBound(60, 90, 110)).toBe('optimal');
    expect(classifyByUpperBound(100, 90, 110)).toBe('caution');
    expect(classifyByUpperBound(120, 90, 110)).toBe('suboptimal');
  });

  it('classifyByTargetRange: a middle band is optimal, both tails degrade', () => {
    expect(classifyByTargetRange(50, 40, 60, 30, 70)).toBe('optimal');
    expect(classifyByTargetRange(35, 40, 60, 30, 70)).toBe('caution');
    expect(classifyByTargetRange(65, 40, 60, 30, 70)).toBe('caution');
    expect(classifyByTargetRange(20, 40, 60, 30, 70)).toBe('suboptimal');
    expect(classifyByTargetRange(80, 40, 60, 30, 70)).toBe('suboptimal');
  });
});

describe('knee drive direction (research-corrected: smaller angle = better front-side mechanics)', () => {
  it('a tightly folded knee (small angle) scores optimal at max velocity', () => {
    const t = PHASE_THRESHOLDS.maxVelocity.kneeDrive;
    expect(classifyByUpperBound(t.optimalMax - 10, t.optimalMax, t.cautionMax)).toBe('optimal');
  });

  it('an open, barely-folded knee (large angle) scores suboptimal', () => {
    const t = PHASE_THRESHOLDS.maxVelocity.kneeDrive;
    expect(classifyByUpperBound(t.cautionMax + 10, t.optimalMax, t.cautionMax)).toBe('suboptimal');
  });

  it('the optimal ceiling gets progressively stricter (smaller) from acceleration to max velocity', () => {
    expect(PHASE_THRESHOLDS.acceleration.kneeDrive.optimalMax).toBeGreaterThan(
      PHASE_THRESHOLDS.transition.kneeDrive.optimalMax
    );
    expect(PHASE_THRESHOLDS.transition.kneeDrive.optimalMax).toBeGreaterThan(
      PHASE_THRESHOLDS.maxVelocity.kneeDrive.optimalMax
    );
  });
});

describe('gait event peak detection', () => {
  const s = (angle: number, t: number) => ({ angle, timestampSeconds: t });

  it('isLocalPeak requires the middle sample to exceed both neighbors by at least minProminence', () => {
    expect(isLocalPeak([s(0, 0), s(10, 1), s(0, 2)], null, 5, 0.1)).toBe(true);
    expect(isLocalPeak([s(0, 0), s(2, 1), s(0, 2)], null, 5, 0.1)).toBe(false);
  });

  it('isLocalPeak respects the refractory period', () => {
    expect(isLocalPeak([s(0, 1.0), s(10, 1.05), s(0, 1.1)], 1.0, 5, 0.12)).toBe(false);
    expect(isLocalPeak([s(0, 1.0), s(10, 1.2), s(0, 1.3)], 1.0, 5, 0.12)).toBe(true);
  });

  it('isGroundContactPeak / isToeOffPeak wrap isLocalPeak with their own tuned constants', () => {
    expect(isGroundContactPeak([s(0.6, 0), s(0.75, 0.1), s(0.6, 0.2)], null)).toBe(true);
    expect(isToeOffPeak([s(140, 0), s(170, 0.1), s(150, 0.2)], null)).toBe(true);
  });

  it('isStepPeak fires on a real knee-angle oscillation peak', () => {
    expect(isStepPeak([s(90, 0), s(130, 0.1), s(95, 0.2)], null)).toBe(true);
  });
});

describe('calculateOverstrideDistance', () => {
  it('measures the horizontal hip-center-to-ankle gap in meters', () => {
    const world = new Array(33).fill(null).map(() => landmark(0, 0));
    world[POSE_LANDMARK_INDICES.LEFT_HIP] = landmark(0, 0, 0);
    world[POSE_LANDMARK_INDICES.RIGHT_HIP] = landmark(0, 0, 0);
    world[POSE_LANDMARK_INDICES.LEFT_ANKLE] = landmark(0.32, 0, 0);
    expect(calculateOverstrideDistance(world, 'left')).toBeCloseTo(0.32, 9);
  });

  it('returns null if a required landmark is unreliable', () => {
    const world = new Array(33).fill(null).map(() => landmark(0, 0));
    world[POSE_LANDMARK_INDICES.LEFT_HIP] = landmark(0, 0, 0);
    world[POSE_LANDMARK_INDICES.RIGHT_HIP] = landmark(0, 0, 0);
    world[POSE_LANDMARK_INDICES.LEFT_ANKLE] = landmark(0.32, 0, 0, 0.1);
    expect(calculateOverstrideDistance(world, 'left')).toBeNull();
  });
});

describe('computeFrameMetrics', () => {
  function makePose({ leftKneeDeg, rightKneeDeg }: { leftKneeDeg: number; rightKneeDeg: number }): PoseLandmarks {
    const arr = basePose();
    arr[POSE_LANDMARK_INDICES.LEFT_SHOULDER] = landmark(0.47, 0.3);
    arr[POSE_LANDMARK_INDICES.RIGHT_SHOULDER] = landmark(0.53, 0.3);
    arr[POSE_LANDMARK_INDICES.LEFT_HIP] = landmark(0.47, 0.5);
    arr[POSE_LANDMARK_INDICES.RIGHT_HIP] = landmark(0.53, 0.5);

    function buildLeg(hipX: number, hipY: number, kneeIdx: number, ankleIdx: number, kneeDeg: number, kneeHigh: boolean) {
      const kneeY = kneeHigh ? 0.35 : 0.62;
      const knee = { x: hipX + 0.05, y: kneeY };
      arr[kneeIdx] = landmark(knee.x, knee.y);
      const hipToKneeAngle = Math.atan2(knee.y - hipY, knee.x - hipX);
      const rad = (kneeDeg * Math.PI) / 180;
      const ankleDir = hipToKneeAngle + (Math.PI - rad);
      const shin = 0.15;
      arr[ankleIdx] = landmark(knee.x + shin * Math.cos(ankleDir), knee.y + shin * Math.sin(ankleDir));
    }
    buildLeg(0.47, 0.5, POSE_LANDMARK_INDICES.LEFT_KNEE, POSE_LANDMARK_INDICES.LEFT_ANKLE, leftKneeDeg, true);
    buildLeg(0.53, 0.5, POSE_LANDMARK_INDICES.RIGHT_KNEE, POSE_LANDMARK_INDICES.RIGHT_ANKLE, rightKneeDeg, false);
    for (const [shoulderIdx, elbowIdx, wristIdx] of [
      [POSE_LANDMARK_INDICES.LEFT_SHOULDER, POSE_LANDMARK_INDICES.LEFT_ELBOW, POSE_LANDMARK_INDICES.LEFT_WRIST],
      [POSE_LANDMARK_INDICES.RIGHT_SHOULDER, POSE_LANDMARK_INDICES.RIGHT_ELBOW, POSE_LANDMARK_INDICES.RIGHT_WRIST],
    ]) {
      const sh = arr[shoulderIdx];
      const elbow = { x: sh.x, y: sh.y + 0.12 };
      arr[elbowIdx] = landmark(elbow.x, elbow.y);
      arr[wristIdx] = landmark(elbow.x + 0.12, elbow.y);
    }
    return arr;
  }

  it('exposes leftKneeAngle/rightKneeAngle unconditionally, regardless of lead/trail role', () => {
    const pose = makePose({ leftKneeDeg: 100, rightKneeDeg: 160 });
    const metrics = computeFrameMetrics(pose, pose, 0, 0, 'maxVelocity');
    expect(metrics.leftKneeAngle).toBeCloseTo(100, 0);
    expect(metrics.rightKneeAngle).toBeCloseTo(160, 0);
  });

  it('assigns kneeDriveAngle/Side to whichever leg is leading (smaller image-y)', () => {
    const pose = makePose({ leftKneeDeg: 100, rightKneeDeg: 160 });
    const metrics = computeFrameMetrics(pose, pose, 0, 0, 'maxVelocity');
    expect(metrics.kneeDriveSide).toBe('left');
    expect(metrics.kneeDriveAngle).toBeCloseTo(100, 0);
  });

  it('leaves an angle null (not a misleading number) when its landmarks are unreliable', () => {
    const pose = makePose({ leftKneeDeg: 100, rightKneeDeg: 160 });
    pose[POSE_LANDMARK_INDICES.LEFT_HIP] = { ...pose[POSE_LANDMARK_INDICES.LEFT_HIP], visibility: 0.1 };
    const metrics = computeFrameMetrics(pose, pose, 0, 0, 'maxVelocity');
    expect(metrics.leftKneeAngle).toBeNull();
  });
});

describe('computePhaseSummaries', () => {
  it('gates event-based metrics behind MIN_GAIT_EVENTS_FOR_SUMMARY', () => {
    const agg = createPhaseAggregates();
    addSampleInPlace(agg.acceleration.kneeExtensionAtToeOff, 100);
    let summaries = computePhaseSummaries(agg);
    // Below the gate, there's no continuous data either, so the phase isn't
    // included in the result at all (see computePhaseSummaries' hasAnyData doc).
    expect(summaries.acceleration).toBeUndefined();

    for (let i = 0; i < MIN_GAIT_EVENTS_FOR_SUMMARY - 1; i++) {
      addSampleInPlace(agg.acceleration.kneeExtensionAtToeOff, 100);
    }
    summaries = computePhaseSummaries(agg);
    expect(summaries.acceleration?.kneeExtensionAtToeOff).not.toBeNull();
  });

  it('excludes event-based metrics from overallScore even when terrible', () => {
    const agg = createPhaseAggregates();
    for (let i = 0; i < MIN_SAMPLES_FOR_SUMMARY + 5; i++) {
      addSampleInPlace(agg.maxVelocity.trunkLean, 5);
      addSampleInPlace(agg.maxVelocity.kneeDrive, 80);
      addSampleInPlace(agg.maxVelocity.hipExtension, 145);
      addSampleInPlace(agg.maxVelocity.leftArmSwing, 90);
      addSampleInPlace(agg.maxVelocity.rightArmSwing, 90);
    }
    for (let i = 0; i < MIN_GAIT_EVENTS_FOR_SUMMARY; i++) {
      addSampleInPlace(agg.maxVelocity.groundContactTime, 0.3); // terrible
      addSampleInPlace(agg.maxVelocity.overstride, 0.9); // terrible
    }
    const summary = computePhaseSummaries(agg).maxVelocity;
    expect(summary?.overallScore).toBeGreaterThan(90);
    expect(summary?.recommendations.some((r) => r.id === 'ground-contact-time')).toBe(true);
    expect(summary?.recommendations.some((r) => r.id === 'overstride')).toBe(true);
  });

  it('includes a phase with gait-event data even if it never reached MIN_SAMPLES_FOR_SUMMARY continuous frames', () => {
    const agg = createPhaseAggregates();
    for (let i = 0; i < MIN_GAIT_EVENTS_FOR_SUMMARY; i++) {
      addSampleInPlace(agg.transition.overstride, 0.1);
    }
    const summaries = computePhaseSummaries(agg);
    expect(summaries.transition).not.toBeUndefined();
  });

  it('never includes a phase with genuinely zero data of any kind', () => {
    const agg = createPhaseAggregates();
    const summaries = computePhaseSummaries(agg);
    expect(summaries.acceleration).toBeUndefined();
    expect(summaries.transition).toBeUndefined();
    expect(summaries.maxVelocity).toBeUndefined();
  });
});

describe('generateCSV', () => {
  it('produces a header row plus one row per frame', () => {
    const frames = [
      {
        frameIndex: 0,
        timestampSeconds: 0,
        phase: 'maxVelocity' as const,
        trunkLeanAngle: 5,
        trunkLeanStatus: 'optimal' as const,
        kneeDriveAngle: 85,
        kneeDriveSide: 'left' as const,
        kneeDriveStatus: 'optimal' as const,
        hipExtensionAngle: 145,
        hipExtensionSide: 'right' as const,
        hipExtensionStatus: 'optimal' as const,
        leftArmSwingAngle: 90,
        leftArmSwingStatus: 'optimal' as const,
        rightArmSwingAngle: 90,
        rightArmSwingStatus: 'optimal' as const,
        leftKneeAngle: 85,
        rightKneeAngle: 160,
      },
    ];
    const csv = generateCSV(frames);
    const lines = csv.split('\n');
    // First line is a measurement-basis comment, second is the header, then one row per frame.
    expect(lines[1]).toContain('frame_index');
    expect(lines[2]).toContain('maxVelocity');
    expect(lines.length).toBe(3);
  });
});

describe('plausibility bounds are internally consistent (min < max)', () => {
  it('cadence', () => {
    expect(CADENCE_MIN_HZ).toBeLessThan(CADENCE_MAX_HZ);
  });
  it('ground contact time', () => {
    expect(GROUND_CONTACT_TIME_MIN_SECONDS).toBeLessThan(GROUND_CONTACT_TIME_MAX_SECONDS);
  });
  it('flight time', () => {
    expect(FLIGHT_TIME_MIN_SECONDS).toBeLessThan(FLIGHT_TIME_MAX_SECONDS);
  });
});

describe('generateCSV formatting details', () => {
  it('formats frame_index as a clean integer, not a decimal', () => {
    const frames = [
      {
        frameIndex: 7,
        timestampSeconds: 1.5,
        phase: 'acceleration' as const,
        trunkLeanAngle: null,
        trunkLeanStatus: null,
        kneeDriveAngle: null,
        kneeDriveSide: null,
        kneeDriveStatus: null,
        hipExtensionAngle: null,
        hipExtensionSide: null,
        hipExtensionStatus: null,
        leftArmSwingAngle: null,
        leftArmSwingStatus: null,
        rightArmSwingAngle: null,
        rightArmSwingStatus: null,
        leftKneeAngle: null,
        rightKneeAngle: null,
      },
    ];
    const csv = generateCSV(frames);
    const dataRow = csv.split('\n')[2];
    expect(dataRow.startsWith('7,')).toBe(true);
    expect(dataRow.startsWith('7.00')).toBe(false);
  });
});

/**
 * Every PHASE_THRESHOLDS entry, checked programmatically rather than by eye:
 * correct internal ordering (an "optimal" band must actually be more
 * demanding than its "caution" band, not overlapping or inverted) and the
 * cross-phase direction each metric's own doc comment claims. A one-off
 * manual reading of these numbers is exactly the kind of check that's easy
 * to get right once and then silently break on a future edit -- this makes
 * it a standing regression guard instead.
 */
describe('PHASE_THRESHOLDS self-consistency', () => {
  it('trunkLean: cautionMin <= optimalMin <= optimalMax <= cautionMax, in every phase', () => {
    for (const phase of SPRINT_PHASES) {
      const t = PHASE_THRESHOLDS[phase].trunkLean;
      expect(t.cautionMin).toBeLessThanOrEqual(t.optimalMin);
      expect(t.optimalMin).toBeLessThanOrEqual(t.optimalMax);
      expect(t.optimalMax).toBeLessThanOrEqual(t.cautionMax);
    }
  });

  it('armSwing: cautionMin <= optimalMin <= optimalMax <= cautionMax, in every phase', () => {
    for (const phase of SPRINT_PHASES) {
      const t = PHASE_THRESHOLDS[phase].armSwing;
      expect(t.cautionMin).toBeLessThanOrEqual(t.optimalMin);
      expect(t.optimalMin).toBeLessThanOrEqual(t.optimalMax);
      expect(t.optimalMax).toBeLessThanOrEqual(t.cautionMax);
    }
  });

  it('lower-bound metrics (hipExtension, kneeExtensionAtToeOff): optimalMin > cautionMin, in every phase', () => {
    for (const phase of SPRINT_PHASES) {
      expect(PHASE_THRESHOLDS[phase].hipExtension.optimalMin).toBeGreaterThan(
        PHASE_THRESHOLDS[phase].hipExtension.cautionMin
      );
      expect(PHASE_THRESHOLDS[phase].kneeExtensionAtToeOff.optimalMin).toBeGreaterThan(
        PHASE_THRESHOLDS[phase].kneeExtensionAtToeOff.cautionMin
      );
    }
  });

  it('upper-bound metrics (kneeDrive, groundContactTime): optimalMax < cautionMax, in every phase', () => {
    for (const phase of SPRINT_PHASES) {
      expect(PHASE_THRESHOLDS[phase].kneeDrive.optimalMax).toBeLessThan(
        PHASE_THRESHOLDS[phase].kneeDrive.cautionMax
      );
      expect(PHASE_THRESHOLDS[phase].groundContactTime.optimalMax).toBeLessThan(
        PHASE_THRESHOLDS[phase].groundContactTime.cautionMax
      );
    }
  });

  it('hipExtension and kneeExtensionAtToeOff targets decrease acceleration -> transition -> maxVelocity', () => {
    const hip = SPRINT_PHASES.map((p) => PHASE_THRESHOLDS[p].hipExtension.optimalMin);
    const knee = SPRINT_PHASES.map((p) => PHASE_THRESHOLDS[p].kneeExtensionAtToeOff.optimalMin);
    expect(hip[0]).toBeGreaterThan(hip[1]);
    expect(hip[1]).toBeGreaterThan(hip[2]);
    expect(knee[0]).toBeGreaterThan(knee[1]);
    expect(knee[1]).toBeGreaterThan(knee[2]);
  });

  it('kneeDrive and groundContactTime targets get stricter (smaller) acceleration -> transition -> maxVelocity', () => {
    const kneeDrive = SPRINT_PHASES.map((p) => PHASE_THRESHOLDS[p].kneeDrive.optimalMax);
    const gct = SPRINT_PHASES.map((p) => PHASE_THRESHOLDS[p].groundContactTime.optimalMax);
    expect(kneeDrive[0]).toBeGreaterThan(kneeDrive[1]);
    expect(kneeDrive[1]).toBeGreaterThan(kneeDrive[2]);
    expect(gct[0]).toBeGreaterThan(gct[1]);
    expect(gct[1]).toBeGreaterThan(gct[2]);
  });

  it('all threshold numbers are finite and positive (ground contact time) or non-negative (angles)', () => {
    for (const phase of SPRINT_PHASES) {
      const t = PHASE_THRESHOLDS[phase];
      const allNumbers = [
        ...Object.values(t.trunkLean),
        ...Object.values(t.kneeDrive),
        ...Object.values(t.hipExtension),
        ...Object.values(t.armSwing),
        ...Object.values(t.kneeExtensionAtToeOff),
        ...Object.values(t.groundContactTime),
      ];
      for (const value of allNumbers) {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeGreaterThanOrEqual(0);
      }
    }
  });
});

describe('CSV phase-summary unit labeling', () => {
  it('every metric row key carries an explicit, correct unit suffix -- no bare "average_angle_deg" column claiming a unit for rows that are actually meters or seconds', () => {
    const agg = createPhaseAggregates();
    for (let i = 0; i < MIN_SAMPLES_FOR_SUMMARY + 5; i++) {
      addSampleInPlace(agg.maxVelocity.trunkLean, 5);
      addSampleInPlace(agg.maxVelocity.kneeDrive, 85);
      addSampleInPlace(agg.maxVelocity.hipExtension, 145);
      addSampleInPlace(agg.maxVelocity.leftArmSwing, 90);
      addSampleInPlace(agg.maxVelocity.rightArmSwing, 90);
    }
    for (let i = 0; i < MIN_GAIT_EVENTS_FOR_SUMMARY; i++) {
      addSampleInPlace(agg.maxVelocity.overstride, 0.1);
      addSampleInPlace(agg.maxVelocity.groundContactTime, 0.09);
      addSampleInPlace(agg.maxVelocity.kneeExtensionAtToeOff, 155);
    }
    const summaries = computePhaseSummaries(agg);
    const csv = generateCSV([], summaries);

    expect(csv).not.toContain('average_angle_deg');
    expect(csv).toContain('trunk_lean_deg');
    expect(csv).toContain('knee_drive_deg');
    expect(csv).toContain('overstride_m');
    expect(csv).toContain('ground_contact_time_s');
  });
});
