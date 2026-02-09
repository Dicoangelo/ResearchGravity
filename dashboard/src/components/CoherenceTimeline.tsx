import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { CoherenceMoment } from "../types";
import { typeColor } from "../theme";

interface Props {
  moments: CoherenceMoment[];
  onSelect: (m: CoherenceMoment) => void;
  selectedId: string | null;
}

export default function CoherenceTimeline({ moments, onSelect, selectedId }: Props) {
  const data = moments.map((m) => ({
    ...m,
    time: new Date(m.created_at).getTime(),
    conf: m.confidence,
  }));

  return (
    <div className="glass-card p-4 h-full">
      <h3 className="text-sm font-medium text-gray-400 mb-3">
        Coherence Timeline
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <ScatterChart margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
          <XAxis
            dataKey="time"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(t) => {
              const d = new Date(t);
              return `${d.getMonth() + 1}/${d.getDate()}`;
            }}
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            dataKey="conf"
            domain={[0, 1]}
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={42}
          />
          <Tooltip
            content={({ payload }) => {
              if (!payload?.[0]) return null;
              const m = payload[0].payload as CoherenceMoment & { time: number };
              return (
                <div className="bg-surface-700 border border-white/10 rounded-lg p-3 text-sm max-w-xs">
                  <p className="font-medium">{m.coherence_type.replace("_", " ")}</p>
                  <p className="text-gray-400 text-xs mt-1">
                    {m.platforms.join(" <-> ")}
                  </p>
                  <p className="text-gray-300 mt-1">
                    {(m.description ?? "").slice(0, 120)}
                  </p>
                  <p className="text-cyan-400 text-xs mt-1">
                    {Math.round(m.confidence * 100)}% confidence
                  </p>
                </div>
              );
            }}
          />
          <Scatter data={data} onClick={(d) => onSelect(d as unknown as CoherenceMoment)}>
            {data.map((d) => (
              <Cell
                key={d.moment_id}
                fill={typeColor(d.coherence_type)}
                fillOpacity={d.moment_id === selectedId ? 1 : 0.7}
                stroke={d.moment_id === selectedId ? "#fff" : "none"}
                strokeWidth={d.moment_id === selectedId ? 2 : 0}
                r={d.moment_id === selectedId ? 7 : 5}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex gap-4 mt-2 text-xs text-gray-500 justify-center">
        {Object.entries({ signature_match: "Signature", semantic_echo: "Semantic", synchronicity: "Synchronicity" }).map(
          ([k, v]) => (
            <span key={k} className="flex items-center gap-1">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full"
                style={{ background: typeColor(k) }}
              />
              {v}
            </span>
          ),
        )}
      </div>
    </div>
  );
}
