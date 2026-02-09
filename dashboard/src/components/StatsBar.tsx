import { Activity, Brain, Layers, Cpu, Zap } from "lucide-react";
import type { OverviewData } from "../types";

interface Props {
  data: OverviewData | null;
}

const stats = (d: OverviewData) => [
  {
    label: "Total Events",
    value: d.platforms.reduce((s, p) => s + p.total, 0).toLocaleString(),
    icon: Activity,
    accent: "#00d9ff",
  },
  {
    label: "Coherence Moments",
    value: d.moments_total.toLocaleString(),
    icon: Brain,
    accent: "#00ff88",
  },
  {
    label: "Moments (24h)",
    value: d.moments_24h.toLocaleString(),
    icon: Zap,
    accent: "#e94560",
  },
  {
    label: "Platforms",
    value: d.platforms.length.toString(),
    icon: Layers,
    accent: "#a78bfa",
  },
  {
    label: "Embeddings",
    value: d.embedded_count.toLocaleString(),
    icon: Cpu,
    accent: "#f59e0b",
  },
];

export default function StatsBar({ data }: Props) {
  if (!data) return null;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 stagger-children">
      {stats(data).map((s) => (
        <div
          key={s.label}
          className="stat-card px-4 py-3 flex items-center gap-3"
          style={{ "--accent-color": `${s.accent}55` } as React.CSSProperties}
        >
          <s.icon
            className="w-4 h-4 shrink-0 opacity-60"
            style={{ color: s.accent }}
          />
          <div className="min-w-0">
            <p className="text-[10px] text-gray-500 uppercase tracking-wider truncate">
              {s.label}
            </p>
            <p
              className="text-lg font-semibold tracking-tight font-mono"
              style={{ color: s.accent }}
            >
              {s.value}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}
