import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import type { NetworkData } from "../types";
import { platformColor, COLORS } from "../theme";

interface Props {
  data: NetworkData | null;
}

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  event_count: number;
  label: string;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  count: number;
  avg_confidence: number;
}

export default function PlatformNetwork({ data }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [nodes, setNodes] = useState<SimNode[]>([]);
  const [links, setLinks] = useState<SimLink[]>([]);

  useEffect(() => {
    if (!data || !svgRef.current) return;

    const width = svgRef.current.clientWidth || 500;
    const height = 280;
    const maxEvents = Math.max(...data.nodes.map((n) => n.event_count), 1);

    const simNodes: SimNode[] = data.nodes.map((n) => ({
      ...n,
      x: width / 2,
      y: height / 2,
    }));

    const simLinks: SimLink[] = data.links.map((l) => ({
      source: l.source,
      target: l.target,
      count: l.count,
      avg_confidence: l.avg_confidence,
    }));

    const radiusScale = d3.scaleSqrt().domain([0, maxEvents]).range([14, 40]);

    const pad = 50;
    const sim = d3
      .forceSimulation(simNodes)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>(simLinks)
          .id((d) => d.id)
          .distance(80),
      )
      .force("charge", d3.forceManyBody().strength(-150))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("x", d3.forceX(width / 2).strength(0.1))
      .force("y", d3.forceY(height / 2).strength(0.1))
      .force("collide", d3.forceCollide<SimNode>().radius((d) => radiusScale(d.event_count) + 8));

    // Run to completion, clamping positions each tick
    for (let i = 0; i < 300; i++) {
      sim.tick();
      simNodes.forEach((n) => {
        n.x = Math.max(pad, Math.min(width - pad, n.x!));
        n.y = Math.max(pad, Math.min(height - pad, n.y!));
      });
    }
    sim.stop();
    setNodes([...simNodes]);
    setLinks([...simLinks]);

    return () => { sim.stop(); };
  }, [data]);

  if (!data) return null;

  const maxEvents = Math.max(...data.nodes.map((n) => n.event_count), 1);
  const maxCount = Math.max(...data.links.map((l) => l.count), 1);
  const radiusScale = d3.scaleSqrt().domain([0, maxEvents]).range([14, 40]);

  return (
    <div className="glass-card p-4 h-full">
      <h3 className="text-sm font-medium text-gray-400 mb-3">
        Platform Network
      </h3>
      <svg ref={svgRef} width="100%" height={280} className="overflow-visible">
        <defs>
          {data.nodes.map((n) => (
            <radialGradient key={n.id} id={`grad-${n.id}`}>
              <stop offset="0%" stopColor={platformColor(n.id)} stopOpacity={0.8} />
              <stop offset="100%" stopColor={platformColor(n.id)} stopOpacity={0.2} />
            </radialGradient>
          ))}
        </defs>

        {/* Links */}
        {links.map((l, i) => {
          const s = l.source as SimNode;
          const t = l.target as SimNode;
          if (s.x == null || t.x == null) return null;
          const opacity = 0.3 + (l.count / maxCount) * 0.5;
          const width = 1 + (l.count / maxCount) * 4;
          return (
            <g key={i}>
              <line
                x1={s.x}
                y1={s.y}
                x2={t.x}
                y2={t.y}
                stroke={COLORS.cyan}
                strokeOpacity={opacity}
                strokeWidth={width}
              />
              <text
                x={(s.x! + t.x!) / 2}
                y={(s.y! + t.y!) / 2 - 6}
                fill="#6b7280"
                fontSize={10}
                textAnchor="middle"
              >
                {l.count}
              </text>
            </g>
          );
        })}

        {/* Nodes */}
        {nodes.map((n) => {
          const r = radiusScale(n.event_count);
          return (
            <g key={n.id}>
              <circle
                cx={n.x}
                cy={n.y}
                r={r}
                fill={`url(#grad-${n.id})`}
                stroke={platformColor(n.id)}
                strokeWidth={1.5}
                strokeOpacity={0.5}
              />
              <text
                x={n.x}
                y={n.y! + r + 14}
                fill="#9ca3af"
                fontSize={11}
                textAnchor="middle"
                fontWeight={500}
              >
                {n.label}
              </text>
              <text
                x={n.x}
                y={n.y! + 4}
                fill="white"
                fontSize={10}
                textAnchor="middle"
                fontWeight={600}
              >
                {n.event_count >= 1000
                  ? `${(n.event_count / 1000).toFixed(0)}K`
                  : n.event_count}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
