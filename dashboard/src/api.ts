import type {
  OverviewData,
  MomentsResponse,
  NetworkData,
  PulseData,
  SignalData,
} from "./types";

const BASE = import.meta.env.VITE_API_URL ?? "";

async function get<T>(path: string, timeoutMs = 15_000): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${BASE}${path}`, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
    return res.json();
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  overview: () => get<OverviewData>("/api/v2/coherence/overview", 30_000),
  moments: (params?: { limit?: number; since_hours?: number; min_confidence?: number }) => {
    const p = new URLSearchParams();
    if (params?.limit) p.set("limit", String(params.limit));
    if (params?.since_hours) p.set("since_hours", String(params.since_hours));
    if (params?.min_confidence) p.set("min_confidence", String(params.min_confidence));
    const qs = p.toString();
    return get<MomentsResponse>(`/api/v2/coherence/moments${qs ? `?${qs}` : ""}`);
  },
  network: () => get<NetworkData>("/api/v2/coherence/network"),
  pulse: (hours = 24) => get<PulseData>(`/api/v2/coherence/pulse?hours=${hours}`, 30_000),
  signals: (momentId: string) =>
    get<SignalData>(`/api/v2/coherence/moment/${momentId}/signals`),
};
