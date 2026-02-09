export interface PlatformStat {
  platform: string;
  total: number;
  today: number;
  last_seen: string | null;
}

export interface CoherenceTypeBreakdown {
  coherence_type: string;
  count: number;
  avg_confidence: number;
}

export interface ConfidenceDistribution {
  tier_90: number;
  tier_80: number;
  tier_70: number;
  tier_60: number;
  tier_low: number;
}

export interface OverviewData {
  platforms: PlatformStat[];
  moments_total: number;
  moments_24h: number;
  by_type: CoherenceTypeBreakdown[];
  confidence_distribution: ConfidenceDistribution;
  embedded_count: number;
}

export interface Signals {
  temporal: number;
  semantic: number;
  meta_cognitive: number;
  instinct_alignment: number;
  concept_overlap: number;
}

export interface CoherenceMoment {
  moment_id: string;
  detected_at: string;
  event_ids: string[];
  platforms: string[];
  coherence_type: string;
  confidence: number;
  description: string;
  time_window_s: number;
  signals: Signals;
  created_at: string;
}

export interface MomentsResponse {
  moments: CoherenceMoment[];
  total: number;
}

export interface NetworkNode {
  id: string;
  event_count: number;
  label: string;
}

export interface NetworkLink {
  source: string;
  target: string;
  count: number;
  avg_confidence: number;
  max_confidence: number;
}

export interface NetworkData {
  nodes: NetworkNode[];
  links: NetworkLink[];
}

export interface PulseActivity {
  platform: string;
  bucket: string;
  event_count: number;
}

export interface PulseArc {
  moment_id: string;
  platforms: string[];
  confidence: number;
  coherence_type: string;
  detected_at: string;
  created_at: string;
}

export interface PulseData {
  activity: PulseActivity[];
  arcs: PulseArc[];
}

export interface SignalData {
  moment_id: string;
  coherence_type: string;
  confidence: number;
  description: string;
  platforms: string[];
  signals: Signals;
}
