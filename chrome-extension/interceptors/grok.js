/**
 * UCW Sovereign Capture — Grok Interceptor v0.3.0
 *
 * Captures messages from grok.x.ai
 * DOM selectors from enhanced-grok-export (Feb 2026, Tailwind CSS):
 *   - User messages:  div.message-row.items-end
 *   - AI messages:    div.message-row.items-start
 *   - Main wrapper:   body > div.flex > div
 *   - No data-testid attributes (confirmed)
 */

/* global UCW */

(function () {
  "use strict";

  const PLATFORM = "grok";
  const seenMessages = new WeakSet();

  // Primary: confirmed Tailwind-based selectors, then fallbacks
  const MESSAGE_SELECTORS = [
    "div.message-row",
    "[class*='message-row']",
    "[class*='MessageRow']",
    "[class*='message-bubble']",
  ].join(", ");

  function extractSessionId() {
    const match = window.location.pathname.match(
      /\/(?:chat|conversation)\/([a-f0-9-]+)/
    );
    return match ? `grok-${match[1]}` : null;
  }

  function extractTitle() {
    // Grok shows conversation title in sidebar or header
    const titleEl =
      document.querySelector("[class*='conversation-title']") ||
      document.querySelector("h1");
    return titleEl?.textContent?.trim() || "";
  }

  function detectDirection(el) {
    // Primary: Tailwind alignment classes from enhanced-grok-export
    // items-end = user (right-aligned), items-start = AI (left-aligned)
    let node = el;
    for (let i = 0; i < 6 && node; i++) {
      const cls = (node.className || "").toLowerCase();

      // Tailwind alignment (most reliable for Grok)
      if (cls.includes("items-end")) return "out";
      if (cls.includes("items-start")) return "in";

      // Semantic class fallbacks
      if (cls.includes("user") || cls.includes("human") || cls.includes("query"))
        return "out";
      if (cls.includes("assistant") || cls.includes("ai") || cls.includes("response") || cls.includes("model"))
        return "in";

      node = node.parentElement;
    }

    return "in";
  }

  function extractContent(el) {
    // Grok wraps response content in .prose or .markdown
    const contentEl = el.querySelector(".prose, .markdown, [class*='message-content']");
    return (contentEl || el).innerText?.trim() || "";
  }

  function processMessages() {
    const messages = document.querySelectorAll(MESSAGE_SELECTORS);
    for (const msg of messages) {
      if (seenMessages.has(msg)) continue;
      seenMessages.add(msg);

      const content = extractContent(msg);
      if (!content || content.length < 5) continue;

      const direction = detectDirection(msg);

      UCW.captureEvent(PLATFORM, content, direction, {
        session_hint: extractSessionId(),
        topic: extractTitle(),
      });
    }
  }

  function init() {
    UCW.observeChat(document.body, processMessages, 500);
    console.log("[UCW] Grok interceptor active");
  }

  if (document.readyState === "complete") init();
  else window.addEventListener("load", init);
})();
