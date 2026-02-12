/**
 * UCW Sovereign Capture — Popup UI Controller
 */

const DEFAULT_SERVER = "http://localhost:3847/api/v2/coherence/capture/extension";

// ── Load state ─────────────────────────────────────

async function loadState() {
  // Get status from background
  chrome.runtime.sendMessage({ type: "UCW_STATUS" }, (response) => {
    if (response) {
      document.getElementById("capture-count").textContent = response.captureCount || 0;
      document.getElementById("queue-size").textContent = response.queueSize || 0;
    }
  });

  // Get config from storage
  const config = await chrome.storage.local.get({
    serverUrl: DEFAULT_SERVER,
    captureEnabled: true,
    platforms: {
      chatgpt: true,
      grok: true,
      gemini: true,
      notebooklm: true,
      youtube: true,
    },
  });

  document.getElementById("server-url").value = config.serverUrl;

  // Set platform toggles
  for (const [platform, enabled] of Object.entries(config.platforms)) {
    const el = document.getElementById(`toggle-${platform}`);
    if (el) el.checked = enabled;
  }

  // Set status
  const statusEl = document.getElementById("status");
  const toggleBtn = document.getElementById("btn-toggle");
  if (config.captureEnabled) {
    statusEl.textContent = "Active";
    statusEl.className = "status active";
    toggleBtn.textContent = "Pause";
  } else {
    statusEl.textContent = "Paused";
    statusEl.className = "status paused";
    toggleBtn.textContent = "Resume";
  }
}

// ── Event Handlers ─────────────────────────────────

document.getElementById("btn-toggle").addEventListener("click", async () => {
  const config = await chrome.storage.local.get({ captureEnabled: true });
  const newState = !config.captureEnabled;
  await chrome.storage.local.set({ captureEnabled: newState });
  loadState();
});

document.getElementById("btn-flush").addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "UCW_FLUSH" }, () => {
    loadState();
  });
});

document.getElementById("server-url").addEventListener("change", async (e) => {
  await chrome.storage.local.set({ serverUrl: e.target.value });
});

// Platform toggles
for (const platform of ["chatgpt", "grok", "gemini", "notebooklm", "youtube"]) {
  const el = document.getElementById(`toggle-${platform}`);
  if (el) {
    el.addEventListener("change", async () => {
      const config = await chrome.storage.local.get({
        platforms: { chatgpt: true, grok: true, gemini: true, notebooklm: true, youtube: true },
      });
      config.platforms[platform] = el.checked;
      await chrome.storage.local.set({ platforms: config.platforms });
    });
  }
}

// ── Init ───────────────────────────────────────────

loadState();
