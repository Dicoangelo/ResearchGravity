/**
 * UCW Sovereign Capture — Background Service Worker
 *
 * Receives captured events from content scripts and sends them
 * to the UCW server capture endpoint.
 */

const DEFAULT_SERVER_URL = "http://localhost:3847/api/v2/coherence/capture/extension";
const BATCH_INTERVAL_MS = 5000; // Flush batch every 5 seconds
const MAX_BATCH_SIZE = 20;

let eventQueue = [];
let batchTimer = null;

// ── Configuration ──────────────────────────────────

async function getConfig() {
  const result = await chrome.storage.local.get({
    serverUrl: DEFAULT_SERVER_URL,
    captureEnabled: true,
    platforms: {
      chatgpt: true,
      grok: true,
      gemini: true,
      notebooklm: true,
      youtube: true,
    },
  });
  return result;
}

// ── Event Queue ────────────────────────────────────

function enqueueEvent(event) {
  eventQueue.push(event);

  if (eventQueue.length >= MAX_BATCH_SIZE) {
    flushBatch();
  } else if (!batchTimer) {
    batchTimer = setTimeout(flushBatch, BATCH_INTERVAL_MS);
  }
}

async function flushBatch() {
  if (batchTimer) {
    clearTimeout(batchTimer);
    batchTimer = null;
  }

  if (eventQueue.length === 0) return;

  const batch = eventQueue.splice(0, MAX_BATCH_SIZE);
  const config = await getConfig();

  try {
    const response = await fetch(config.serverUrl + "/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(batch),
    });

    if (!response.ok) {
      console.error("[UCW] Batch send failed:", response.status);
      // Re-queue failed events (up to a limit)
      if (eventQueue.length < 100) {
        eventQueue.unshift(...batch);
      }
    } else {
      const result = await response.json();
      console.log(`[UCW] Batch sent: ${result.captured} captured, ${result.errors} errors`);

      // Update badge
      updateBadge(result.captured);
    }
  } catch (err) {
    console.error("[UCW] Batch send error:", err.message);
    // Re-queue on network failure
    if (eventQueue.length < 100) {
      eventQueue.unshift(...batch);
    }
  }
}

// ── Badge ──────────────────────────────────────────

let captureCount = 0;

function updateBadge(count) {
  captureCount += count;
  chrome.action.setBadgeText({ text: String(captureCount) });
  chrome.action.setBadgeBackgroundColor({ color: "#6366f1" });
}

// ── Message Listener ───────────────────────────────

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "UCW_CAPTURE") {
    getConfig().then((config) => {
      if (!config.captureEnabled) {
        sendResponse({ status: "disabled" });
        return;
      }

      const platform = message.platform;
      if (config.platforms[platform] === false) {
        sendResponse({ status: "platform_disabled" });
        return;
      }

      enqueueEvent(message.event);
      sendResponse({ status: "queued", queueSize: eventQueue.length });
    });
    return true; // Keep message channel open for async response
  }

  if (message.type === "UCW_STATUS") {
    sendResponse({
      queueSize: eventQueue.length,
      captureCount,
      enabled: true,
    });
    return false;
  }

  if (message.type === "UCW_FLUSH") {
    flushBatch().then(() => {
      sendResponse({ status: "flushed" });
    });
    return true;
  }
});

// ── Startup ────────────────────────────────────────

chrome.runtime.onInstalled.addListener(() => {
  console.log("[UCW] Sovereign Capture extension installed");
  chrome.action.setBadgeText({ text: "0" });
  chrome.action.setBadgeBackgroundColor({ color: "#6366f1" });
});
