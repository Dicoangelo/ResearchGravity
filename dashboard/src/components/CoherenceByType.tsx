import type { CoherenceTypeBreakdown } from "../types";
import { typeColor } from "../theme";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import { Layers } from "lucide-react";

interface Props {
  data: CoherenceTypeBreakdown[] | null;
}

const TYPE_LABELS: Record<string, string> = {
  signature_match: "Signature Match",
  semantic_echo: "Semantic Echo",
  synchronicity: "Synchronicity",
};

function typeLabel(type: string): string {
  return TYPE_LABELS[type] ?? type;
}

export default function CoherenceByType({ data }: Props) {
  if (!data || data.length === 0) return null;

  const total = data.reduce((sum, d) => sum + d.count, 0);

  const chartData = data.map((d) => ({
    name: typeLabel(d.coherence_type),
    value: d.count,
    avgConfidence: d.avg_confidence,
    fill: typeColor(d.coherence_type),
  }));

  return (
    <div className="glass-card p-5">
      <h3 className="text-xs font-semibold tracking-wider uppercase text-gray-500 mb-4 flex items-center gap-2">
        <Layers size={14} className="text-cyan-400" />
        Moments by Type
        <span className="text-[10px] font-mono text-gray-600 ml-auto">
          {total} total
        </span>
      </h3>

      <div className="flex items-center gap-6">
        {/* Donut chart */}
        <div className="w-32 h-32 flex-shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={32}
                outerRadius={56}
                paddingAngle={3}
                dataKey="value"
                stroke="none"
              >
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} opacity={0.85} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: "#0f1524",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: "8px",
                  fontSize: "11px",
                }}
                formatter={(value: number, _name: string, props: { payload?: { avgConfidence?: number } }) => {
                  const conf = props?.payload?.avgConfidence ?? 0;
                  return [
                    `${value} (${Math.round((value / total) * 100)}%) — avg ${Math.round(conf * 100)}%`,
                    "",
                  ];
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Legend + stats */}
        <div className="flex-1 space-y-3">
          {data.map((d) => {
            const pct = Math.round((d.count / total) * 100);
            const color = typeColor(d.coherence_type);
            return (
              <div key={d.coherence_type}>
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-2.5 h-2.5 rounded-sm"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-[11px] text-gray-300">
                      {typeLabel(d.coherence_type)}
                    </span>
                  </div>
                  <span className="text-[11px] font-mono text-gray-400">
                    {d.count}
                    <span className="text-gray-600 ml-1">({pct}%)</span>
                  </span>
                </div>
                <div className="flex items-center gap-2 ml-[18px]">
                  <div className="flex-1 h-1 bg-white/5 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.round(d.avg_confidence * 100)}%`,
                        backgroundColor: `${color}88`,
                      }}
                    />
                  </div>
                  <span className="text-[9px] font-mono text-gray-500">
                    {Math.round(d.avg_confidence * 100)}% conf
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
