import { describe, expect, it } from 'vitest';
import {
  downsampleChartRows,
  overstrideSteps,
  phaseDurations,
  toChartRows,
  toPhaseBands,
  type ChartRow,
  type GaitEventRecord,
} from './chartData';
import type { FrameMetrics, SprintPhase } from './biomechanics';

function frame(overrides: Partial<FrameMetrics> = {}): FrameMetrics {
  return {
    frameIndex: 0,
    timestampSeconds: 0,
    phase: 'acceleration',
    trunkLeanAngle: 30,
    trunkLeanStatus: 'optimal',
    kneeDriveAngle: 90,
    kneeDriveSide: 'left',
    kneeDriveStatus: 'optimal',
    hipExtensionAngle: 160,
    hipExtensionSide: 'left',
    hipExtensionStatus: 'optimal',
    leftArmSwingAngle: 100,
    leftArmSwingStatus: 'optimal',
    rightArmSwingAngle: 100,
    rightArmSwingStatus: 'optimal',
    leftKneeAngle: 90,
    rightKneeAngle: 90,
    ...overrides,
  };
}

function gaitEvent(overrides: Partial<GaitEventRecord> = {}): GaitEventRecord {
  return {
    type: 'groundContact',
    side: 'left',
    timestampSeconds: 0,
    value: 0.1,
    phase: 'acceleration',
    ...overrides,
  };
}

describe('toChartRows', () => {
  it('maps FrameMetrics fields onto the flatter chart-row shape', () => {
    const rows = toChartRows([frame({ timestampSeconds: 1.5, trunkLeanAngle: 22 })]);
    expect(rows).toEqual<ChartRow[]>([
      {
        t: 1.5,
        trunkLean: 22,
        kneeDrive: 90,
        hipExtension: 160,
        leftArmSwing: 100,
        rightArmSwing: 100,
        leftKnee: 90,
        rightKnee: 90,
        phase: 'acceleration',
      },
    ]);
  });

  it('preserves nulls rather than coercing them', () => {
    const rows = toChartRows([frame({ trunkLeanAngle: null, kneeDriveAngle: null, kneeDriveSide: null })]);
    expect(rows[0].trunkLean).toBeNull();
    expect(rows[0].kneeDrive).toBeNull();
  });
});

describe('downsampleChartRows', () => {
  it('returns the input unchanged when already at or below maxPoints', () => {
    const rows = toChartRows([frame({ timestampSeconds: 0 }), frame({ timestampSeconds: 1 })]);
    expect(downsampleChartRows(rows, 10)).toEqual(rows);
  });

  it('bucket-averages numeric fields and keeps the bucket-start timestamp', () => {
    const rows: ChartRow[] = [
      { t: 0, trunkLean: 10, kneeDrive: 80, hipExtension: 150, leftArmSwing: 90, rightArmSwing: 90, leftKnee: 80, rightKnee: 80, phase: 'acceleration' },
      { t: 1, trunkLean: 20, kneeDrive: 100, hipExtension: 170, leftArmSwing: 110, rightArmSwing: 110, leftKnee: 100, rightKnee: 100, phase: 'acceleration' },
    ];
    const result = downsampleChartRows(rows, 1);
    expect(result).toHaveLength(1);
    expect(result[0].t).toBe(0);
    expect(result[0].trunkLean).toBe(15);
    expect(result[0].kneeDrive).toBe(90);
  });

  it('averages only non-null samples within a bucket, not null-as-zero', () => {
    const rows: ChartRow[] = [
      { t: 0, trunkLean: 10, kneeDrive: null, hipExtension: null, leftArmSwing: null, rightArmSwing: null, leftKnee: null, rightKnee: null, phase: 'acceleration' },
      { t: 1, trunkLean: 30, kneeDrive: null, hipExtension: null, leftArmSwing: null, rightArmSwing: null, leftKnee: null, rightKnee: null, phase: 'acceleration' },
    ];
    const result = downsampleChartRows(rows, 1);
    expect(result[0].trunkLean).toBe(20);
    expect(result[0].kneeDrive).toBeNull();
  });
});

describe('toPhaseBands', () => {
  it('collapses consecutive same-phase frames into a single band', () => {
    const frames = [
      frame({ phase: 'acceleration', timestampSeconds: 0 }),
      frame({ phase: 'acceleration', timestampSeconds: 1 }),
      frame({ phase: 'acceleration', timestampSeconds: 2 }),
    ];
    expect(toPhaseBands(frames)).toEqual([{ phase: 'acceleration', startTime: 0, endTime: 2 }]);
  });

  it('starts a new band on every phase transition, in order', () => {
    const frames = [
      frame({ phase: 'acceleration', timestampSeconds: 0 }),
      frame({ phase: 'transition', timestampSeconds: 1 }),
      frame({ phase: 'transition', timestampSeconds: 2 }),
      frame({ phase: 'maxVelocity', timestampSeconds: 3 }),
    ];
    expect(toPhaseBands(frames)).toEqual([
      { phase: 'acceleration', startTime: 0, endTime: 0 },
      { phase: 'transition', startTime: 1, endTime: 2 },
      { phase: 'maxVelocity', startTime: 3, endTime: 3 },
    ]);
  });

  it('returns an empty array for no frames', () => {
    expect(toPhaseBands([])).toEqual([]);
  });

  it('re-opens a band for a phase seen earlier but not immediately prior (no incorrect merging)', () => {
    const frames = [
      frame({ phase: 'acceleration', timestampSeconds: 0 }),
      frame({ phase: 'transition', timestampSeconds: 1 }),
      frame({ phase: 'acceleration', timestampSeconds: 2 }),
    ];
    expect(toPhaseBands(frames)).toEqual([
      { phase: 'acceleration', startTime: 0, endTime: 0 },
      { phase: 'transition', startTime: 1, endTime: 1 },
      { phase: 'acceleration', startTime: 2, endTime: 2 },
    ]);
  });
});

describe('phaseDurations', () => {
  it('sums band durations per phase, including across multiple bands of the same phase', () => {
    const durations = phaseDurations([
      { phase: 'acceleration', startTime: 0, endTime: 2 },
      { phase: 'transition', startTime: 2, endTime: 3 },
      { phase: 'acceleration', startTime: 5, endTime: 6 },
    ]);
    expect(durations).toEqual<Partial<Record<SprintPhase, number>>>({
      acceleration: 3,
      transition: 1,
    });
  });

  it('returns an empty object for no bands', () => {
    expect(phaseDurations([])).toEqual({});
  });
});

describe('overstrideSteps', () => {
  it('keeps only groundContact events with a non-null value, sorted chronologically', () => {
    const events: GaitEventRecord[] = [
      gaitEvent({ type: 'groundContact', timestampSeconds: 2, value: 0.2 }),
      gaitEvent({ type: 'toeOff', timestampSeconds: 1, value: 170 }),
      gaitEvent({ type: 'groundContact', timestampSeconds: 0.5, value: 0.1 }),
      gaitEvent({ type: 'groundContact', timestampSeconds: 1.5, value: null }),
    ];
    expect(overstrideSteps(events)).toEqual([
      { timestampSeconds: 0.5, value: 0.1 },
      { timestampSeconds: 2, value: 0.2 },
    ]);
  });

  it('returns an empty array when there are no groundContact events', () => {
    expect(overstrideSteps([gaitEvent({ type: 'toeOff' })])).toEqual([]);
  });
});
