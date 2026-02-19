import type { Breakthrough } from "../types";
import { platformLabel, platformColor } from "../theme";
import { Zap, TrendingUp, Atom } from "lucide-react";

interface Props {
  breakthroughs: Breakthrough[];
}

const TYPE_CONFIG: Record<string, { icon: typeof Zap; color: string; label: string }> = {
  synthesis: { icon: Atom, color: "#a78bfa", label: "Synthesis" },
  convergence: { icon: TrendingUp, color: "#00ff88", label: "Convergence" },
  emergence: { icon: Zap, color: "#e94560", label: "Emergence" },
};

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[9px] text-gray-500 w-10">{label}</span>
      <div className="flex-1 h-1 bg-white/5 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{ width: `${value * 100}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-[9px] font-mono" style={{ color }}>
        {Math.round(value * 100)}
      </span>
    </div>
  );
}

export default function BreakthroughsFeed({ breakthroughs }: Props) {
  const sorted = [...breakthroughs]
    .sort((a, b) => new Date(b.detected_at).getTime() - new Date(a.detected_at).getTime())
    .slice(0, 8);

  return (
    <div className="glass-card p-5">
      <h3 className="text-xs font-semibold tracking-wider uppercase text-gray-500 mb-4 flex items-center gap-2">
        <Zap size={14} className="text-amber-400" />
        Breakthroughs
        <span className="text-[10px] font-mono text-gray-600 ml-auto">
          {breakthroughs.length} detected
        </span>
      </h3>

      {sorted.length === 0 ? (
        <div className="flex items-center justify-center h-32 text-gray-600 text-sm">
          No breakthroughs detected yet
        </div>
      ) : (
        <div className="space-y-3 max-h-[420px] overflow-y-auto pr-1">
          {sorted.map((bt) => {
            const cfg = TYPE_CONFIG[bt.type] ?? TYPE_CONFIG.synthesis;
            const Icon = cfg.icon;
            const age = formatAge(bt.detected_at);

            return (
              <div
                key={bt.breakthrough_id}
                className="bg-white/[0.02] border border-white/[0.04] rounded-xl p-3.5 hover:border-white/[0.08] transition-colors"
              >
                {/* Header */}
                <div className="flex items-start gap-2 mb-2">
                  <div
                    className="w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
                    style={{ backgroundColor: `${cfg.color}15`, border: `1px solid ${cfg.color}25` }}
                  >
                    <Icon size={12} style={{ color: cfg.color }} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-[12px] font-medium text-gray-200 leading-tight truncate">
                      {bt.title}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <span
                        className="text-[9px] font-semibold tracking-wider uppercase"
                        style={{ color: cfg.color }}
                      >
                        {cfg.label}
                      </span>
                      <span className="text-[9px] text-gray-600">{age}</span>
                    </div>
                  </div>
                </div>

                {/* Narrative */}
                <p className="text-[11px] text-gray-400 leading-relaxed mb-2.5 line-clamp-2">
                  {bt.narrative}
                </p>

                {/* Scores */}
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 mb-2">
                  <ScoreBar label="Novel" value={bt.novelty_score} color="#f59e0b" />
                  <ScoreBar label="Impact" value={bt.impact_score} color="#e94560" />
                </div>

                {/* Concepts + Platforms */}
                <div className="flex items-center gap-1.5 flex-wrap">
                  {bt.concepts.map((c) => (
                    <span
                      key={c}
                      className="text-[9px] px-1.5 py-0.5 rounded bg-cyan-400/5 text-cyan-400/70 border border-cyan-400/10"
                    >
                      {c}
                    </span>
                  ))}
                  <span className="flex-1" />
                  {bt.platforms.map((p) => (
                    <span
                      key={p}
                      className="text-[9px] font-mono"
                      style={{ color: platformColor(p) }}
                    >
                      {platformLabel(p)}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function formatAge(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const hours = Math.floor(diff / 3_600_000);
  if (hours < 1) return "just now";
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days === 1) return "1d ago";
  return `${days}d ago`;
}
