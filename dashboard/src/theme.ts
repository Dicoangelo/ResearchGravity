export const COLORS = {
  bg: "#060a13",
  bgGrad: "#0d1525",
  card: "#0f1524",
  cardHover: "#151d30",
  border: "rgba(255, 255, 255, 0.06)",
  accent: "#e94560",
  cyan: "#00d9ff",
  green: "#00ff88",
  purple: "#a78bfa",
  amethyst: "#9333ea",
  amber: "#f59e0b",
  gold: "#fbbf24",
  text: "#e5e7eb",
  textMuted: "#6b7280",
  textDim: "#374151",
  // Semantic layer colors
  dataLayer: "#00d9ff",    // What was said
  lightLayer: "#a78bfa",   // What it means
  instinctLayer: "#e94560", // What it signals
} as const;

export const PLATFORM_COLORS: Record<string, string> = {
  "claude-cli": "#00d9ff",
  chatgpt: "#00ff88",
  "claude-code": "#a78bfa",
  "claude-desktop": "#e94560",
  ccc: "#f59e0b",
  grok: "#8b5cf6",
  test: "#374151",
};

export const TYPE_COLORS: Record<string, string> = {
  signature_match: "#00ff88",
  semantic_echo: "#00d9ff",
  synchronicity: "#e94560",
};

export function platformColor(platform: string): string {
  return PLATFORM_COLORS[platform] ?? "#4b5563";
}

export function typeColor(type: string): string {
  return TYPE_COLORS[type] ?? "#4b5563";
}

export const PLATFORM_LABELS: Record<string, string> = {
  "claude-cli": "CLI",
  chatgpt: "ChatGPT",
  "claude-code": "Code",
  "claude-desktop": "Desktop",
  ccc: "CCC",
  grok: "Grok",
  test: "Test",
};

export function platformLabel(platform: string): string {
  return PLATFORM_LABELS[platform] ?? platform;
}
