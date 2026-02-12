/**
 * UCW Sovereign Capture — Shared capture utilities for content scripts.
 *
 * Each platform interceptor calls captureEvent() to send data
 * to the background service worker.
 */

/**
 * Send a captured event to the background service worker.
 *
 * @param {string} platform - Platform name (chatgpt, grok, gemini, etc.)
 * @param {string} content - The message content
 * @param {string} direction - "in" (assistant) or "out" (user)
 * @param {object} opts - Optional fields: topic, intent, concepts, session_hint, url
 */
function captureEvent(platform, content, direction, opts = {}) {
  if (!content || content.trim().length < 5) return;

  const event = {
    platform,
    content: content.substring(0, 10000), // Cap at 10k chars
    direction,
    url: opts.url || window.location.href,
    topic: opts.topic || "",
    intent: opts.intent || "",
    concepts: opts.concepts || [],
    session_hint: opts.session_hint || null,
    metadata: opts.metadata || {},
  };

  chrome.runtime.sendMessage(
    { type: "UCW_CAPTURE", platform, event },
    (response) => {
      if (chrome.runtime.lastError) {
        console.warn("[UCW] Capture failed:", chrome.runtime.lastError.message);
        return;
      }
      if (response?.status === "queued") {
        console.debug(`[UCW] Captured ${direction} on ${platform} (queue: ${response.queueSize})`);
      }
    }
  );
}

/**
 * Set up a MutationObserver to watch for new messages in a chat container.
 *
 * @param {string} containerSelector - CSS selector for the chat container
 * @param {string} messageSelector - CSS selector for individual messages
 * @param {function} extractMessage - (element) => { content, direction } or null to skip
 * @param {string} platform - Platform name
 */
function observeChat(containerSelector, messageSelector, extractMessage, platform) {
  const seenMessages = new WeakSet();

  function processNewMessages() {
    const messages = document.querySelectorAll(messageSelector);
    for (const msg of messages) {
      if (seenMessages.has(msg)) continue;
      seenMessages.add(msg);

      const extracted = extractMessage(msg);
      if (extracted && extracted.content) {
        captureEvent(platform, extracted.content, extracted.direction, extracted.opts || {});
      }
    }
  }

  // Initial scan
  processNewMessages();

  // Watch for new messages
  const target = document.querySelector(containerSelector) || document.body;
  const observer = new MutationObserver(() => {
    processNewMessages();
  });

  observer.observe(target, {
    childList: true,
    subtree: true,
  });

  console.log(`[UCW] Observing ${platform} chat`);
  return observer;
}
