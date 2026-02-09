import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { ConfidenceDistribution as ConfDist } from "../types";

interface Props {
  data: ConfDist | null;
}

const TIERS = [
  { key: "tier_90", label: "90-100%", color: "#00ff88" },
  { key: "tier_80", label: "80-89%", color: "#22d3ee" },
  { key: "tier_70", label: "70-79%", color: "#a78bfa" },
  { key: "tier_60", label: "60-69%", color: "#f59e0b" },
  { key: "tier_low", label: "<60%", color: "#ef4444" },
] as const;

export default function ConfidenceDistribution({ data }: Props) {
  if (!data) return null;

  const chartData = TIERS.map((t) => ({
    name: t.label,
    value: data[t.key as keyof ConfDist] ?? 0,
    color: t.color,
  }));

  return (
    <div className="glass-card p-4 h-full">
      <h3 className="text-sm font-medium text-gray-400 mb-3">
        Confidence Distribution
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 10 }}>
          <XAxis type="number" hide />
          <YAxis
            type="category"
            dataKey="name"
            width={65}
            tick={{ fill: "#9ca3af", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              background: "#1a1f35",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 8,
              color: "#e5e7eb",
            }}
            formatter={(v: number) => [`${v} moments`, "Count"]}
          />
          <Bar dataKey="value" radius={[0, 6, 6, 0]} maxBarSize={28}>
            {chartData.map((d, i) => (
              <Cell key={i} fill={d.color} fillOpacity={0.8} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
