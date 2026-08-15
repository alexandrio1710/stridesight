/**
 * StrideSight — Chart Data Transforms
 *
 * Pure functions that reshape the existing `FrameMetrics[]` / gait-event
 * history the analyzer already produces into the row/series shapes Recharts
 * wants. No DOM, no React, no charting-library imports — mirrors
 * `biomechanics.ts`'s "fully unit-testable in isolation" convention so this
 * stays easy to test without a browser environment.
 *
 * `GaitEventRecord` is defined here (not in biomechanics.ts) because it's
 * produced by VideoAnalyzer's own gait-event detection loop, not by the
 * biomechanics engine itself — biomechanics.ts's aggregates only keep
 * running statistics, not a timestamped history of individual events, so
 * this is a separate, additive capture used purely for charting.
 */

import type { FrameMetrics, Side, SprintPhase } from './biomechanics';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GaitEventRecord {
  type: 'groundContact' | 'toeOff';
  side: Side;
  timestampSeconds: number;
  /** Overstride distance (meters) for a groundContact event, knee-extension angle (degrees) for a toeOff event. */
  value: number | null;
  phase: SprintPhase;
}

/** One row per analyzed frame, flattened to the fields the charts plot. */
export interface ChartRow {
  t: number;
  trunkLean: number | null;
  kneeDrive: number | null;
  hipExtension: number | null;
  leftArmSwing: number | null;
  rightArmSwing: number | null;
  leftKnee: number | null;
  rightKnee: number | null;
  phase: SprintPhase;
}

export interface PhaseBand {
  phase: SprintPhase;
  startTime: number;
  endTime: number;
}

// ---------------------------------------------------------------------------
// Frame -> chart row
// ---------------------------------------------------------------------------

export function toChartRows(frames: FrameMetrics[]): ChartRow[] {
  return frames.map((f) => ({
    t: f.timestampSeconds,
    trunkLean: f.trunkLeanAngle,
    kneeDrive: f.kneeDriveAngle,
    hipExtension: f.hipExtensionAngle,
    leftArmSwing: f.leftArmSwingAngle,
    rightArmSwing: f.rightArmSwingAngle,
    leftKnee: f.leftKneeAngle,
    rightKnee: f.rightKneeAngle,
    phase: f.phase,
  }));
}

// ---------------------------------------------------------------------------
// Downsampling — bucket-averages numeric fields so Recharts stays smooth on
// long clips. Phase bands are computed separately from the *full-resolution*
// frames (see toPhaseBands below) so a short phase can't get smeared away by
// bucket averaging.
// ---------------------------------------------------------------------------

const numericKeys = [
  'trunkLean',
  'kneeDrive',
  'hipExtension',
  'leftArmSwing',
  'rightArmSwing',
  'leftKnee',
  'rightKnee',
] as const satisfies readonly (keyof ChartRow)[];

function averageBucket(bucket: ChartRow[]): ChartRow {
  const last = bucket[bucket.length - 1];
  const row: ChartRow = { ...last, t: bucket[0].t };

  for (const key of numericKeys) {
    let sum = 0;
    let count = 0;
    for (const r of bucket) {
      const value = r[key];
      if (value !== null) {
        sum += value;
        count += 1;
      }
    }
    row[key] = count > 0 ? sum / count : null;
  }

  return row;
}

export function downsampleChartRows(rows: ChartRow[], maxPoints = 300): ChartRow[] {
  if (rows.length <= maxPoints) return rows;

  const bucketSize = Math.ceil(rows.length / maxPoints);
  const result: ChartRow[] = [];

  for (let i = 0; i < rows.length; i += bucketSize) {
    result.push(averageBucket(rows.slice(i, i + bucketSize)));
  }

  return result;
}

// ---------------------------------------------------------------------------
// Phase bands — collapse consecutive same-phase frames into segments for
// Recharts' ReferenceArea shading.
// ---------------------------------------------------------------------------

export function toPhaseBands(frames: FrameMetrics[]): PhaseBand[] {
  const bands: PhaseBand[] = [];

  for (const frame of frames) {
    const current = bands[bands.length - 1];
    if (current && current.phase === frame.phase) {
      current.endTime = frame.timestampSeconds;
    } else {
      bands.push({ phase: frame.phase, startTime: frame.timestampSeconds, endTime: frame.timestampSeconds });
    }
  }

  return bands;
}

/** Total time spent in each phase, derived from the same bands used for chart shading. */
export function phaseDurations(bands: PhaseBand[]): Partial<Record<SprintPhase, number>> {
  const durations: Partial<Record<SprintPhase, number>> = {};
  for (const band of bands) {
    const duration = band.endTime - band.startTime;
    durations[band.phase] = (durations[band.phase] ?? 0) + duration;
  }
  return durations;
}

// ---------------------------------------------------------------------------
// Gait events
// ---------------------------------------------------------------------------

/** Ground-contact events with a measured overstride value, in chronological order — the series a per-step bar chart plots. */
export function overstrideSteps(events: GaitEventRecord[]): { timestampSeconds: number; value: number }[] {
  return events
    .filter((e): e is GaitEventRecord & { value: number } => e.type === 'groundContact' && e.value !== null)
    .map((e) => ({ timestampSeconds: e.timestampSeconds, value: e.value }))
    .sort((a, b) => a.timestampSeconds - b.timestampSeconds);
}
