/**
 * UCW Sovereign Capture — Gemini Interceptor v0.3.0
 *
 * Captures messages from gemini.google.com
 * Gemini DOM structure (Feb 2026):
 *   - Custom elements: <model-response>, <user-query> (if present)
 *   - Content: .markdown-main-panel, .response-content
 *   - URL: /app/{id} or /chats/{id}
 *   - Fallbacks for Angular/Lit component changes
 */

/* global UCW */

(function () {
  "use strict";

  const PLATFORM = "gemini";
  const seenMessages = new WeakSet();

  // Custom elements first, then class-based fallbacks, then role-based
  const MESSAGE_SELECTORS = [
    "model-response",
    "user-query",
    "[class*='query-text']",
    "[class*='response-container']",
    "[role='article']",
    "[role='listitem']",
    ".conversation-container > div",
  ].join(", ");

  function extractSessionId() {
    // Gemini URLs: /app/{id}, /chat/{id}, or /chats/{id}
    const match = window.location.pathname.match(
      /\/(?:app|chat|chats)\/([a-zA-Z0-9_-]+)/
    );
    return match ? `gemini-${match[1]}` : null;
  }

  function extractTitle() {
    // Gemini shows conversation title in sidebar or tab
    const titleEl =
      document.querySelector("[class*='conversation-title']") ||
      document.querySelector("[class*='chat-title']");
    return titleEl?.textContent?.trim() || "";
  }

  function detectDirection(el) {
    // Custom element detection (most reliable)
    const tagName = (el.tagName || "").toLowerCase();
    if (tagName === "user-query") return "out";
    if (tagName === "model-response") return "in";

    // Class-based detection
    const cls = (el.className || "").toLowerCase();
    if (cls.includes("query") || cls.includes("user") || cls.includes("human-turn"))
      return "out";
    if (cls.includes("response") || cls.includes("model") || cls.includes("bot-turn"))
      return "in";

    // Walk up for custom element or class context
    let node = el.parentElement;
    for (let i = 0; i < 5 && node; i++) {
      const tag = (node.tagName || "").toLowerCase();
      if (tag === "user-query") return "out";
      if (tag === "model-response") return "in";

      const parentCls = (node.className || "").toLowerCase();
      if (parentCls.includes("query") || parentCls.includes("user")) return "out";
      if (parentCls.includes("response") || parentCls.includes("model")) return "in";

      node = node.parentElement;
    }

    return "in";
  }

  function extractContent(el) {
    const contentEl = el.querySelector(
      ".markdown-main-panel, .response-content, .query-content, .markdown, .prose"
    );
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
    console.log("[UCW] Gemini interceptor active");
  }

  if (document.readyState === "complete") init();
  else window.addEventListener("load", init);
})();
