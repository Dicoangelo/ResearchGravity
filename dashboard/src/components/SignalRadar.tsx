import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { Signals } from "../types";

interface Props {
  signals: Signals | null;
  label?: string;
}

const AXES: { key: keyof Signals; label: string }[] = [
  { key: "temporal", label: "Temporal" },
  { key: "semantic", label: "Semantic" },
  { key: "meta_cognitive", label: "Meta-Cognitive" },
  { key: "instinct_alignment", label: "Instinct" },
  { key: "concept_overlap", label: "Concept" },
];

export default function SignalRadar({ signals, label }: Props) {
  const data = AXES.map((a) => ({
    axis: a.label,
    value: signals?.[a.key] ?? 0,
    fullMark: 1,
  }));

  return (
    <div className="glass-card p-4 h-full">
      <h3 className="text-sm font-medium text-gray-400 mb-1">
        Signal Breakdown
      </h3>
      {label && (
        <p className="text-xs text-gray-600 mb-2 truncate">{label}</p>
      )}
      <ResponsiveContainer width="100%" height={230}>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="72%">
          <PolarGrid stroke="rgba(255,255,255,0.08)" />
          <PolarAngleAxis
            dataKey="axis"
            tick={{ fill: "#9ca3af", fontSize: 11 }}
          />
          <PolarRadiusAxis
            domain={[0, 1]}
            tickCount={5}
            tick={{ fill: "#4b5563", fontSize: 9 }}
            axisLine={false}
          />
          <Tooltip
            contentStyle={{
              background: "#1a1f35",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 8,
              color: "#e5e7eb",
            }}
            formatter={(v: number) => [`${(v * 100).toFixed(0)}%`, "Signal"]}
          />
          <Radar
            dataKey="value"
            stroke="#e94560"
            fill="#e94560"
            fillOpacity={0.25}
            strokeWidth={2}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
