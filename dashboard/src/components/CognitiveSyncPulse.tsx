import { useMemo } from "react";
import type { PulseData } from "../types";
import { platformColor, platformLabel, typeColor, COLORS } from "../theme";

interface Props {
  data: PulseData | null;
  platforms: string[];
}

const CX = 300;
const CY = 200;

// Three semantic layer rings (inner = Data, middle = Light, outer = Instinct)
const LAYER_RINGS = [
  { r: 35, label: "DATA", color: COLORS.dataLayer, opacity: 0.15 },
  { r: 52, label: "LIGHT", color: COLORS.lightLayer, opacity: 0.1 },
  { r: 68, label: "INSTINCT", color: COLORS.instinctLayer, opacity: 0.08 },
];

// Platform orbits
const ORBIT_RADII = [95, 120, 148, 172, 195, 215, 232];

export default function CognitiveSyncPulse({ data, platforms }: Props) {
  const sorted = useMemo(() => {
    if (!data) return platforms;
    const activity: Record<string, number> = {};
    for (const a of data.activity) {
      activity[a.platform] = (activity[a.platform] ?? 0) + a.event_count;
    }
    return [...platforms].sort(
      (a, b) => (activity[b] ?? 0) - (activity[a] ?? 0),
    );
  }, [platforms, data]);

  const platformMap = useMemo(() => {
    const map: Record<string, { radius: number; angle: number }> = {};
    sorted.forEach((p, i) => {
      map[p] = {
        radius: ORBIT_RADII[i] ?? 220,
        angle: (i / sorted.length) * Math.PI * 2 - Math.PI / 2,
      };
    });
    return map;
  }, [sorted]);

  const platformActivity = useMemo(() => {
    if (!data) return {};
    const acc: Record<string, number> = {};
    for (const a of data.activity) {
      acc[a.platform] = (acc[a.platform] ?? 0) + a.event_count;
    }
    return acc;
  }, [data]);

  const eventDots = useMemo(() => {
    if (!data) return [];
    const dots: {
      cx: number;
      cy: number;
      platform: string;
      opacity: number;
      r: number;
    }[] = [];
    const perPlatform: Record<string, number> = {};

    for (const a of data.activity) {
      const count = perPlatform[a.platform] ?? 0;
      if (count >= 10) continue;
      perPlatform[a.platform] = count + 1;

      const pm = platformMap[a.platform];
      if (!pm) continue;

      const angle = pm.angle + count * 0.35 - 1.4;
      const r = pm.radius + (Math.sin(count * 1.7) * 8);
      dots.push({
        cx: CX + Math.cos(angle) * r,
        cy: CY + Math.sin(angle) * r,
        platform: a.platform,
        opacity: 0.2 + Math.min(a.event_count / 40, 0.6),
        r: 2 + Math.min(a.event_count / 30, 2.5),
      });
    }
    return dots;
  }, [data, platformMap]);

  const arcs = useMemo(() => {
    if (!data) return [];
    return data.arcs
      .filter((a) => a.platforms.length === 2)
      .slice(0, 25)
      .map((a, i) => {
        const pA = platformMap[a.platforms[0]];
        const pB = platformMap[a.platforms[1]];
        if (!pA || !pB) return null;

        const x1 = CX + Math.cos(pA.angle) * pA.radius;
        const y1 = CY + Math.sin(pA.angle) * pA.radius;
        const x2 = CX + Math.cos(pB.angle) * pB.radius;
        const y2 = CY + Math.sin(pB.angle) * pB.radius;

        // Curve through center with varying curvature
        const curveFactor = 0.1 + (i % 3) * 0.08;
        const midX = CX + (x1 + x2 - CX * 2) * curveFactor;
        const midY = CY + (y1 + y2 - CY * 2) * curveFactor;

        return {
          ...a,
          path: `M ${x1} ${y1} Q ${midX} ${midY} ${x2} ${y2}`,
          color: typeColor(a.coherence_type),
          delay: (i * 0.4) % 3,
        };
      })
      .filter(Boolean);
  }, [data, platformMap]);

  if (!sorted.length) return null;

  return (
    <div className="glass-card-glow p-5 relative overflow-hidden">
      {/* Ambient glow behind the card */}
      <div
        className="absolute inset-0 pointer-events-none animate-breathe"
        style={{
          background:
            "radial-gradient(ellipse at 50% 40%, rgba(0,217,255,0.04) 0%, transparent 70%)",
        }}
      />

      <div className="flex items-center justify-between mb-2 relative z-10">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-gray-300 tracking-wide uppercase">
            Cognitive Sync Pulse
          </h3>
          {/* Three semantic layer indicators */}
          <div className="flex items-center gap-2 ml-2">
            {LAYER_RINGS.map((l) => (
              <span
                key={l.label}
                className="text-[9px] font-medium tracking-widest px-1.5 py-0.5 rounded-full border"
                style={{
                  color: l.color,
                  borderColor: `${l.color}33`,
                  background: `${l.color}0a`,
                }}
              >
                {l.label}
              </span>
            ))}
          </div>
        </div>
        {data && (
          <span className="text-xs text-gray-500 font-mono">
            <span className="text-cyan-400/70">{data.arcs.length}</span>{" "}
            coherence arcs
          </span>
        )}
      </div>

      <svg
        viewBox="0 0 600 400"
        className="w-full max-h-[420px] relative z-10"
      >
        <defs>
          {/* Center breathing glow */}
          <radialGradient id="center-glow-2">
            <stop offset="0%" stopColor={COLORS.cyan} stopOpacity={0.4} />
            <stop offset="30%" stopColor={COLORS.cyan} stopOpacity={0.1} />
            <stop offset="60%" stopColor={COLORS.purple} stopOpacity={0.03} />
            <stop offset="100%" stopColor="transparent" stopOpacity={0} />
          </radialGradient>

          {/* Glow filter */}
          <filter id="glow2">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Soft glow for arcs */}
          <filter id="arc-glow">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Platform node gradients */}
          {sorted.map((p) => (
            <radialGradient key={`pgrad-${p}`} id={`pgrad-${p}`}>
              <stop
                offset="0%"
                stopColor={platformColor(p)}
                stopOpacity={0.9}
              />
              <stop
                offset="70%"
                stopColor={platformColor(p)}
                stopOpacity={0.3}
              />
              <stop
                offset="100%"
                stopColor={platformColor(p)}
                stopOpacity={0}
              />
            </radialGradient>
          ))}
        </defs>

        {/* ═══ CENTER: Sovereign core ═══ */}
        <circle
          cx={CX}
          cy={CY}
          r={80}
          fill="url(#center-glow-2)"
          className="animate-breathe"
        />

        {/* Three semantic layer rings */}
        {LAYER_RINGS.map((layer, i) => (
          <g key={layer.label}>
            <circle
              cx={CX}
              cy={CY}
              r={layer.r}
              fill="none"
              stroke={layer.color}
              strokeWidth={0.8}
              strokeOpacity={layer.opacity + 0.08}
              strokeDasharray={i === 0 ? "none" : i === 1 ? "8 4" : "3 5"}
            />
          </g>
        ))}

        {/* Center core */}
        <circle cx={CX} cy={CY} r={10} fill={COLORS.cyan} fillOpacity={0.15} />
        <circle cx={CX} cy={CY} r={5} fill={COLORS.cyan} fillOpacity={0.5} />
        <circle cx={CX} cy={CY} r={2} fill="white" fillOpacity={0.9} />

        {/* ═══ ORBITAL RINGS ═══ */}
        {sorted.map((p, i) => {
          const r = ORBIT_RADII[i] ?? 220;
          const activity = platformActivity[p] ?? 0;
          const intensity = Math.min(activity / 80, 1);
          return (
            <circle
              key={`orbit-${p}`}
              cx={CX}
              cy={CY}
              r={r}
              fill="none"
              stroke={platformColor(p)}
              strokeWidth={0.4 + intensity * 0.6}
              strokeOpacity={0.06 + intensity * 0.12}
              strokeDasharray={intensity > 0.3 ? "none" : "2 8"}
            />
          );
        })}

        {/* ═══ COHERENCE ARCS ═══ */}
        {arcs.map((a) =>
          a ? (
            <path
              key={a.moment_id}
              d={a.path}
              fill="none"
              stroke={a.color}
              strokeWidth={0.8 + a.confidence * 2.5}
              strokeOpacity={0.25 + a.confidence * 0.5}
              strokeLinecap="round"
              filter="url(#arc-glow)"
              className="animate-arc-flow"
              style={{ animationDelay: `${a.delay}s`, animationDuration: `${2.5 + a.delay}s` }}
            >
              <title>
                {a.coherence_type} | {Math.round(a.confidence * 100)}% |{" "}
                {a.platforms.join(" <-> ")}
              </title>
            </path>
          ) : null,
        )}

        {/* ═══ EVENT DOTS ═══ */}
        {eventDots.map((d, i) => (
          <circle
            key={`dot-${i}`}
            cx={d.cx}
            cy={d.cy}
            r={d.r}
            fill={platformColor(d.platform)}
            fillOpacity={d.opacity}
          />
        ))}

        {/* ═══ PLATFORM NODES ═══ */}
        {sorted.map((p, i) => {
          const r = ORBIT_RADII[i] ?? 220;
          const angle = (i / sorted.length) * Math.PI * 2 - Math.PI / 2;
          const px = CX + Math.cos(angle) * r;
          const py = CY + Math.sin(angle) * r;
          const lx = CX + Math.cos(angle) * (r + 22);
          const ly = CY + Math.sin(angle) * (r + 22);
          const activity = platformActivity[p] ?? 0;
          const nodeR = 5 + Math.min(activity / 200, 4);

          return (
            <g key={`node-${p}`}>
              {/* Glow halo */}
              {activity > 0 && (
                <circle
                  cx={px}
                  cy={py}
                  r={nodeR + 6}
                  fill={`url(#pgrad-${p})`}
                  className="animate-breathe"
                  style={{ animationDelay: `${i * 0.6}s` }}
                />
              )}
              {/* Node */}
              <circle
                cx={px}
                cy={py}
                r={nodeR}
                fill={platformColor(p)}
                fillOpacity={0.85}
                stroke={platformColor(p)}
                strokeWidth={activity > 0 ? 1.5 : 0}
                strokeOpacity={0.25}
              />
              {/* Label */}
              <text
                x={lx}
                y={ly}
                fill={platformColor(p)}
                fontSize={10}
                fontWeight={600}
                textAnchor="middle"
                dominantBaseline="middle"
                style={{ letterSpacing: "0.03em" }}
              >
                {platformLabel(p)}
              </text>
              {activity > 0 && (
                <text
                  x={lx}
                  y={ly + 13}
                  fill="#4b5563"
                  fontSize={8}
                  textAnchor="middle"
                  fontFamily="monospace"
                >
                  {activity.toLocaleString()}
                </text>
              )}
            </g>
          );
        })}

        {/* ═══ CENTER LABEL ═══ */}
        <text
          x={CX}
          y={CY + 20}
          fill="#374151"
          fontSize={7}
          textAnchor="middle"
          letterSpacing={3}
          fontWeight={500}
        >
          YOU
        </text>
      </svg>
    </div>
  );
}
