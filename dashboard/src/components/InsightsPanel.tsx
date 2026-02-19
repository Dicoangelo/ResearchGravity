import type { CoherenceMoment } from "../types";
import { typeColor, platformLabel } from "../theme";
import { Lightbulb, Sparkles, Tag } from "lucide-react";

interface Props {
  moment: CoherenceMoment | null;
}

const CATEGORY_LABELS: Record<string, { label: string; color: string }> = {
  crystallization: { label: "Crystallization", color: "#00d9ff" },
  convergence: { label: "Convergence", color: "#a78bfa" },
  emergence: { label: "Emergence", color: "#00ff88" },
  synthesis: { label: "Synthesis", color: "#f59e0b" },
  breakthrough: { label: "Breakthrough", color: "#e94560" },
};

export default function InsightsPanel({ moment }: Props) {
  if (!moment) {
    return (
      <div className="glass-card p-5">
        <h3 className="text-xs font-semibold tracking-wider uppercase text-gray-500 mb-4 flex items-center gap-2">
          <Lightbulb size={14} />
          Insight
        </h3>
        <div className="flex items-center justify-center h-32 text-gray-600 text-sm">
          Select a moment from the timeline
        </div>
      </div>
    );
  }

  const cat = CATEGORY_LABELS[moment.insight_category ?? ""] ?? {
    label: moment.insight_category ?? "Unknown",
    color: "#4b5563",
  };
  const novelty = moment.insight_novelty ?? 0;
  const noveltyPct = Math.round(novelty * 100);
  const hasInsight = !!moment.insight_summary;

  return (
    <div className="glass-card p-5">
      <h3 className="text-xs font-semibold tracking-wider uppercase text-gray-500 mb-3 flex items-center gap-2">
        <Lightbulb size={14} className="text-amber-400" />
        Insight
      </h3>

      {/* Meta row */}
      <div className="flex items-center gap-3 mb-3 flex-wrap">
        {/* Category badge */}
        <span
          className="text-[10px] font-semibold tracking-wider uppercase px-2.5 py-1 rounded-full border"
          style={{
            color: cat.color,
            borderColor: `${cat.color}33`,
            backgroundColor: `${cat.color}0d`,
          }}
        >
          <Tag size={10} className="inline mr-1 -mt-px" />
          {cat.label}
        </span>

        {/* Coherence type */}
        <span
          className="text-[10px] font-mono px-2 py-0.5 rounded"
          style={{
            color: typeColor(moment.coherence_type),
            backgroundColor: `${typeColor(moment.coherence_type)}15`,
          }}
        >
          {moment.coherence_type.replace("_", " ")}
        </span>

        {/* Platforms */}
        <span className="text-[10px] text-gray-500">
          {moment.platforms.map(platformLabel).join(" ↔ ")}
        </span>

        {/* Confidence */}
        <span className="text-[10px] font-mono text-gray-400 ml-auto">
          {Math.round(moment.confidence * 100)}%
        </span>
      </div>

      {/* Novelty bar */}
      <div className="flex items-center gap-2 mb-4">
        <Sparkles size={12} className="text-amber-400/60" />
        <span className="text-[10px] text-gray-500 w-12">Novelty</span>
        <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${noveltyPct}%`,
              background: `linear-gradient(90deg, #f59e0b, #e94560)`,
            }}
          />
        </div>
        <span className="text-[10px] font-mono text-amber-400/80 w-8 text-right">
          {noveltyPct}%
        </span>
      </div>

      {/* Insight text */}
      {hasInsight ? (
        <p className="text-[13px] leading-relaxed text-gray-300/90 max-h-48 overflow-y-auto pr-1">
          {moment.insight_summary}
        </p>
      ) : (
        <p className="text-sm text-gray-600 italic">
          No insight extracted for this moment.
        </p>
      )}
    </div>
  );
}
