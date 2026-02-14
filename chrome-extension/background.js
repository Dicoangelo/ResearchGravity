/**
 * UCW Sovereign Capture — Background Service Worker v0.2.0
 *
 * Receives captured events from content scripts and sends them
 * to the UCW server capture endpoint.
 *
 * Hardened:
 *  - chrome.storage.local queue persistence (survives SW termination)
 *  - Exponential backoff with jitter on failures
 *  - chrome.alarms periodic flush (survives SW restarts)
 *  - Badge count tracking
 */

const DEFAULT_SERVER_URL = "http://localhost:3847/api/v2/coherence/capture/extension";
const BATCH_INTERVAL_MS = 5000;
const MAX_BATCH_SIZE = 20;
const MAX_QUEUE_SIZE = 200;
const BASE_RETRY_MS = 2000;
const MAX_RETRY_MS = 60000;
const ALARM_NAME = "ucw-flush";
const ALARM_PERIOD_MIN = 0.5; // 30 seconds

let eventQueue = [];
let batchTimer = null;
let retryCount = 0;
let configCache = null;

// ── Startup: Recover persisted queue ──────────────────

chrome.runtime.onStartup.addListener(async () => {
  await recoverQueue();
});

chrome.runtime.onInstalled.addListener(async () => {
  console.log("[UCW] Sovereign Capture extension installed");
  chrome.action.setBadgeText({ text: "0" });
  chrome.action.setBadgeBackgroundColor({ color: "#6366f1" });
  await recoverQueue();

  // Set up periodic alarm (survives SW termination)
  chrome.alarms.create(ALARM_NAME, { periodInMinutes: ALARM_PERIOD_MIN });
});

// Alarm-based periodic flush
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === ALARM_NAME) {
    flushBatch();
  }
});

async function recoverQueue() {
  try {
    const { ucwQueue } = await chrome.storage.local.get("ucwQueue");
    if (ucwQueue && ucwQueue.length > 0) {
      eventQueue.unshift(...ucwQueue);
      console.log(`[UCW] Recovered ${ucwQueue.length} queued events from storage`);
      await chrome.storage.local.remove("ucwQueue");
      flushBatch();
    }
  } catch (err) {
    console.error("[UCW] Queue recovery failed:", err.message);
  }
}

async function persistQueue() {
  if (eventQueue.length === 0) {
    await chrome.storage.local.remove("ucwQueue");
    return;
  }
  try {
    // Keep most recent 100 events max
    const toStore = eventQueue.slice(0, 100);
    await chrome.storage.local.set({ ucwQueue: toStore });
  } catch (err) {
    console.error("[UCW] Queue persistence failed:", err.message);
  }
}

// ── Configuration ─────────────────────────────────────

async function getConfig() {
  if (configCache && Date.now() - configCache.ts < 30000) {
    return configCache.data;
  }
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
  configCache = { data: result, ts: Date.now() };
  return result;
}

// ── Event Queue ───────────────────────────────────────

function enqueueEvent(event) {
  if (eventQueue.length >= MAX_QUEUE_SIZE) {
    // Drop oldest to make room
    eventQueue.shift();
  }

  eventQueue.push({
    ...event,
    _enqueuedAt: Date.now(),
  });

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
      requeue(batch);
      scheduleRetry();
      return;
    }

    const result = await response.json();
    console.log(`[UCW] Batch sent: ${result.captured} captured, ${result.errors} errors`);
    updateBadge(result.captured);
    retryCount = 0; // Reset on success

    // Flush more if queue still has items
    if (eventQueue.length > 0) {
      batchTimer = setTimeout(flushBatch, 500);
    }
  } catch (err) {
    console.error("[UCW] Batch send error:", err.message);
    requeue(batch);
    scheduleRetry();
  }
}

function requeue(batch) {
  // Re-add to front if we have room
  const room = MAX_QUEUE_SIZE - eventQueue.length;
  if (room > 0) {
    eventQueue.unshift(...batch.slice(0, room));
  }
  // Persist to survive SW termination
  persistQueue();
}

function scheduleRetry() {
  retryCount++;
  // Exponential backoff with jitter: 2s, 4s, 8s, ... up to 60s
  const delay = Math.min(BASE_RETRY_MS * Math.pow(2, retryCount - 1), MAX_RETRY_MS);
  const jitter = delay * 0.2 * Math.random();
  batchTimer = setTimeout(flushBatch, delay + jitter);
  console.log(`[UCW] Retry #${retryCount} in ${Math.round((delay + jitter) / 1000)}s`);
}

// ── Badge ─────────────────────────────────────────────

let captureCount = 0;

function updateBadge(count) {
  captureCount += count;
  const text = captureCount > 999 ? "999+" : String(captureCount);
  chrome.action.setBadgeText({ text });
  chrome.action.setBadgeBackgroundColor({ color: "#6366f1" });
}

// ── Message Listener ──────────────────────────────────

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
      retryCount,
      enabled: true,
    });
    return false;
  }

  if (message.type === "UCW_FLUSH") {
    flushBatch().then(() => {
      sendResponse({ status: "flushed", remaining: eventQueue.length });
    });
    return true;
  }
});

// Ensure alarm exists (in case onInstalled didn't fire this session)
chrome.alarms.get(ALARM_NAME, (alarm) => {
  if (!alarm) {
    chrome.alarms.create(ALARM_NAME, { periodInMinutes: ALARM_PERIOD_MIN });
  }
});
