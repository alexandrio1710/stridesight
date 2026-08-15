'use client';

import { useMemo, type ReactNode } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  downsampleChartRows,
  overstrideSteps,
  phaseDurations,
  toChartRows,
  toPhaseBands,
  type GaitEventRecord,
} from '@/utils/chartData';
import { PHASE_LABELS, type FrameMetrics, type SprintPhase } from '@/utils/biomechanics';

interface BiomechanicsDashboardProps {
  frameHistory: FrameMetrics[];
  gaitEvents: GaitEventRecord[];
}

/** Same phase-color language used elsewhere in the app (PHASE_BADGE_CLASSES in VideoAnalyzer), as raw hex for SVG fills/strokes. */
const PHASE_COLORS: Record<SprintPhase, string> = {
  acceleration: '#fb923c', // orange-400
  transition: '#c084fc', // purple-400
  maxVelocity: '#38bdf8', // sky-400
};

const GAIT_EVENT_COLORS = {
  groundContact: '#4ade80', // green-400
  toeOff: '#facc15', // yellow-400
};

function formatSeconds(value: number): string {
  return `${value.toFixed(1)}s`;
}

/** Recharts' Tooltip `labelFormatter` passes the axis label as `ReactNode`, not `number` — coerce defensively. */
function formatSecondsLabel(label: ReactNode): string {
  const value = typeof label === 'number' ? label : Number(label);
  return Number.isFinite(value) ? formatSeconds(value) : '';
}

interface ChartCardProps {
  title: string;
  caption: string;
  children: ReactNode;
}

function ChartCard({ title, caption, children }: ChartCardProps): ReactNode {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</h3>
      <p className="mt-0.5 text-[11px] leading-snug text-slate-600">{caption}</p>
      <div className="mt-3 h-64 w-full">{children}</div>
    </div>
  );
}

function PhaseBandsLayer({ bands }: { bands: ReturnType<typeof toPhaseBands> }): ReactNode {
  return (
    <>
      {bands.map((band, i) => (
        <ReferenceArea
          key={i}
          x1={band.startTime}
          x2={band.endTime}
          fill={PHASE_COLORS[band.phase]}
          fillOpacity={0.08}
          stroke="none"
          ifOverflow="extendDomain"
        />
      ))}
    </>
  );
}

function GaitEventMarkers({ events, type }: { events: GaitEventRecord[]; type: GaitEventRecord['type'] }): ReactNode {
  const color = GAIT_EVENT_COLORS[type];
  return (
    <>
      {events
        .filter((e) => e.type === type)
        .map((e, i) => (
          <ReferenceLine
            key={`${type}-${i}`}
            x={e.timestampSeconds}
            stroke={color}
            strokeOpacity={0.5}
            strokeDasharray="2 2"
            ifOverflow="extendDomain"
          />
        ))}
    </>
  );
}

export default function BiomechanicsDashboard({ frameHistory, gaitEvents }: BiomechanicsDashboardProps): ReactNode {
  const rows = useMemo(() => downsampleChartRows(toChartRows(frameHistory)), [frameHistory]);
  const bands = useMemo(() => toPhaseBands(frameHistory), [frameHistory]);
  const durations = useMemo(() => phaseDurations(bands), [bands]);
  const overstrideData = useMemo(() => overstrideSteps(gaitEvents), [gaitEvents]);

  const durationRows = (Object.keys(durations) as SprintPhase[]).map((phase) => ({
    phase: PHASE_LABELS[phase],
    seconds: durations[phase] ?? 0,
    fill: PHASE_COLORS[phase],
  }));

  if (frameHistory.length === 0) return null;

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
      <ChartCard
        title="Knee Angle Over Time"
        caption="Left/right knee-ankle-hip angle across the clip, shaded by detected sprint phase. Dashed green/yellow lines mark ground-contact and toe-off events."
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="t" tickFormatter={formatSeconds} stroke="#64748b" fontSize={11} />
            <YAxis stroke="#64748b" fontSize={11} unit="°" />
            <Tooltip
              contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', fontSize: 12 }}
              labelFormatter={formatSecondsLabel}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <PhaseBandsLayer bands={bands} />
            <GaitEventMarkers events={gaitEvents} type="groundContact" />
            <GaitEventMarkers events={gaitEvents} type="toeOff" />
            <Line type="monotone" dataKey="leftKnee" name="Left Knee" stroke="#38bdf8" dot={false} strokeWidth={1.5} connectNulls />
            <Line type="monotone" dataKey="rightKnee" name="Right Knee" stroke="#f472b6" dot={false} strokeWidth={1.5} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard
        title="Trunk Lean Over Time"
        caption="Forward lean angle against true vertical — the signal StrideSight uses to auto-detect sprint phase."
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="t" tickFormatter={formatSeconds} stroke="#64748b" fontSize={11} />
            <YAxis stroke="#64748b" fontSize={11} unit="°" />
            <Tooltip
              contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', fontSize: 12 }}
              labelFormatter={formatSecondsLabel}
            />
            <PhaseBandsLayer bands={bands} />
            <Line type="monotone" dataKey="trunkLean" name="Trunk Lean" stroke="#fb923c" dot={false} strokeWidth={2} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard
        title="Arm Swing & Hip Extension"
        caption="Elbow angle through the swing (both arms) and trail-leg hip extension at push-off."
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="t" tickFormatter={formatSeconds} stroke="#64748b" fontSize={11} />
            <YAxis stroke="#64748b" fontSize={11} unit="°" />
            <Tooltip
              contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', fontSize: 12 }}
              labelFormatter={formatSecondsLabel}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <PhaseBandsLayer bands={bands} />
            <Line type="monotone" dataKey="leftArmSwing" name="Left Arm" stroke="#38bdf8" dot={false} strokeWidth={1.5} connectNulls />
            <Line type="monotone" dataKey="rightArmSwing" name="Right Arm" stroke="#f472b6" dot={false} strokeWidth={1.5} connectNulls />
            <Line type="monotone" dataKey="hipExtension" name="Hip Extension" stroke="#a3e635" dot={false} strokeWidth={1.5} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard
        title="Over-stride Per Step"
        caption="Hip-to-ankle horizontal gap at each detected ground contact. Smaller is better — landing ahead of the hips brakes forward momentum."
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={overstrideData} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="timestampSeconds" tickFormatter={formatSeconds} stroke="#64748b" fontSize={11} />
            <YAxis stroke="#64748b" fontSize={11} unit="m" />
            <Tooltip
              contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', fontSize: 12 }}
              labelFormatter={formatSecondsLabel}
              formatter={(value) => [`${Number(value).toFixed(2)} m`, 'Over-stride']}
            />
            <Bar dataKey="value" fill="#38bdf8" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {durationRows.length > 0 && (
        <div className="rounded-xl border border-white/10 bg-white/5 p-4 xl:col-span-2">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Phase Duration Breakdown</h3>
          <p className="mt-0.5 text-[11px] leading-snug text-slate-600">Time spent in each detected sprint phase across the whole clip.</p>
          <div className="mt-3 h-32 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={durationRows} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" horizontal={false} />
                <XAxis type="number" tickFormatter={formatSeconds} stroke="#64748b" fontSize={11} />
                <YAxis type="category" dataKey="phase" stroke="#64748b" fontSize={11} width={80} />
                <Tooltip
                  contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', fontSize: 12 }}
                  formatter={(value) => [formatSeconds(Number(value)), 'Duration']}
                />
                <Bar dataKey="seconds" radius={[0, 3, 3, 0]}>
                  {durationRows.map((row, i) => (
                    <Cell key={i} fill={row.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
