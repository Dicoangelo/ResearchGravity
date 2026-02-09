import { useEffect, useState, useCallback } from "react";
import { api } from "./api";
import type {
  OverviewData,
  CoherenceMoment,
  NetworkData,
  PulseData,
  Signals,
} from "./types";
import StatsBar from "./components/StatsBar";
import CognitiveSyncPulse from "./components/CognitiveSyncPulse";
import CoherenceTimeline from "./components/CoherenceTimeline";
import PlatformNetwork from "./components/PlatformNetwork";
import ConfidenceDistribution from "./components/ConfidenceDistribution";
import SignalRadar from "./components/SignalRadar";

const REFRESH_MS = 30_000;

export default function App() {
  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [moments, setMoments] = useState<CoherenceMoment[]>([]);
  const [network, setNetwork] = useState<NetworkData | null>(null);
  const [pulse, setPulse] = useState<PulseData | null>(null);
  const [selectedMoment, setSelectedMoment] =
    useState<CoherenceMoment | null>(null);
  const [selectedSignals, setSelectedSignals] = useState<Signals | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const fetchAll = useCallback(async () => {
    const [ov, mom, net, pul] = await Promise.allSettled([
      api.overview(),
      api.moments({ limit: 200, since_hours: 720 }),
      api.network(),
      api.pulse(24),
    ]);
    if (ov.status === "fulfilled") setOverview(ov.value);
    if (mom.status === "fulfilled") setMoments(mom.value.moments);
    if (net.status === "fulfilled") setNetwork(net.value);
    if (pul.status === "fulfilled") setPulse(pul.value);

    const failed = [ov, mom, net, pul].filter(
      (r) => r.status === "rejected",
    );
    setError(
      failed.length > 0 ? `${failed.length} endpoint(s) loading` : null,
    );
    setLastRefresh(new Date());
  }, []);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, REFRESH_MS);
    return () => clearInterval(interval);
  }, [fetchAll]);

  const handleMomentSelect = useCallback(async (m: CoherenceMoment) => {
    setSelectedMoment(m);
    setSelectedSignals(m.signals);
    try {
      const sig = await api.signals(m.moment_id);
      setSelectedSignals(sig.signals);
    } catch {
      // Fall back to inline signals
    }
  }, []);

  const platforms = overview?.platforms.map((p) => p.platform) ?? [];

  return (
    <div className="min-h-screen p-4 md:p-8 max-w-[1440px] mx-auto">
      {/* ═══ HEADER ═══ */}
      <header className="flex items-end justify-between mb-6 animate-fade-in">
        <div>
          <div className="flex items-center gap-3 mb-1">
            {/* Brand mark */}
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-600/20 border border-cyan-500/20 flex items-center justify-center">
              <svg
                viewBox="0 0 24 24"
                className="w-4 h-4 text-cyan-400"
                fill="none"
                stroke="currentColor"
                strokeWidth={2}
              >
                <circle cx="12" cy="12" r="3" />
                <circle cx="12" cy="12" r="8" opacity={0.4} />
                <path d="M12 2v4M12 18v4M2 12h4M18 12h4" opacity={0.3} />
              </svg>
            </div>
            <h1 className="text-xl font-bold tracking-tight text-white">
              <span className="text-cyan-400">UCW</span>{" "}
              <span className="text-gray-300 font-light">Coherence</span>
            </h1>
          </div>
          <p className="text-[11px] text-gray-600 tracking-wider uppercase ml-11">
            Sovereign Cognitive Infrastructure
          </p>
        </div>

        <div className="flex items-center gap-4 text-xs">
          {error && (
            <span className="text-amber-400/80 bg-amber-400/5 border border-amber-400/10 px-2.5 py-1 rounded-lg font-mono text-[10px]">
              {error}
            </span>
          )}
          <span className="text-gray-600 font-mono text-[10px]">
            {lastRefresh.toLocaleTimeString()}
          </span>
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-30" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-400" />
          </span>
        </div>
      </header>

      {/* ═══ STATS ═══ */}
      <div className="animate-fade-in" style={{ animationDelay: "0.1s" }}>
        <StatsBar data={overview} />
      </div>

      {/* ═══ HERO: Cognitive Sync Pulse ═══ */}
      <div
        className="mt-5 animate-fade-in"
        style={{ animationDelay: "0.2s" }}
      >
        <CognitiveSyncPulse data={pulse} platforms={platforms} />
      </div>

      {/* ═══ CHARTS ROW 1 ═══ */}
      <div
        className="grid grid-cols-1 lg:grid-cols-2 gap-5 mt-5 animate-fade-in"
        style={{ animationDelay: "0.3s" }}
      >
        <CoherenceTimeline
          moments={moments}
          onSelect={handleMomentSelect}
          selectedId={selectedMoment?.moment_id ?? null}
        />
        <PlatformNetwork data={network} />
      </div>

      {/* ═══ CHARTS ROW 2 ═══ */}
      <div
        className="grid grid-cols-1 lg:grid-cols-2 gap-5 mt-5 animate-fade-in"
        style={{ animationDelay: "0.4s" }}
      >
        <ConfidenceDistribution
          data={overview?.confidence_distribution ?? null}
        />
        <SignalRadar
          signals={selectedSignals}
          label={
            selectedMoment
              ? `${selectedMoment.coherence_type} | ${selectedMoment.platforms.join(" <-> ")} | ${Math.round(selectedMoment.confidence * 100)}%`
              : "Select a moment from the timeline"
          }
        />
      </div>

      {/* ═══ FOOTER ═══ */}
      <footer className="mt-8 pb-4 flex items-center justify-center gap-3">
        <div className="h-px flex-1 max-w-[80px] bg-gradient-to-r from-transparent to-gray-800" />
        <p className="text-[10px] text-gray-700 tracking-[0.2em] uppercase font-light">
          Metaventions — Cognitive Equity
        </p>
        <div className="h-px flex-1 max-w-[80px] bg-gradient-to-l from-transparent to-gray-800" />
      </footer>
    </div>
  );
}
