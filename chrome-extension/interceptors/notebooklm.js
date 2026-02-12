/**
 * UCW Sovereign Capture — NotebookLM Interceptor v0.3.0
 *
 * Captures interactions from notebooklm.google.com.
 * DOM selectors sourced from notebooklm-py project (Feb 2026):
 *   - User queries:  .from-user-container
 *   - AI responses:  .to-user-container
 *   - Chat history:  [role='log']
 *   - Notebook URL:  /notebook/{NOTEBOOK_ID}
 */

/* global UCW */

(function () {
  "use strict";

  const PLATFORM = "notebooklm";
  const seenMessages = new WeakSet();

  // Primary selectors from notebooklm-py, then fallbacks
  const MESSAGE_SELECTORS = [
    ".from-user-container",
    ".to-user-container",
    "[role='log'] > div",
    "[class*='chat-message']",
    "[class*='message-container']",
  ].join(", ");

  function extractSessionId() {
    const match = window.location.pathname.match(
      /\/notebook\/([a-zA-Z0-9_-]+)/
    );
    return match ? `notebooklm-${match[1]}` : null;
  }

  function extractNotebookTitle() {
    const titleEl = document.querySelector(
      "[class*='notebook-title'], [class*='NotebookTitle'], h1"
    );
    return titleEl?.textContent?.trim() || "";
  }

  function detectDirection(el) {
    const cls = (el.className || "").toLowerCase();

    // Primary: confirmed class names from notebooklm-py
    if (cls.includes("from-user")) return "out";
    if (cls.includes("to-user")) return "in";

    // Fallback: generic patterns
    const testId = (el.getAttribute("data-testid") || "").toLowerCase();
    const combined = cls + " " + testId;
    if (combined.includes("query") || combined.includes("user")) return "out";
    if (combined.includes("response") || combined.includes("answer") || combined.includes("model"))
      return "in";

    // Walk up parents
    let node = el.parentElement;
    for (let i = 0; i < 4 && node; i++) {
      const parentCls = (node.className || "").toLowerCase();
      if (parentCls.includes("from-user")) return "out";
      if (parentCls.includes("to-user")) return "in";
      if (parentCls.includes("query") || parentCls.includes("user")) return "out";
      if (parentCls.includes("response") || parentCls.includes("answer")) return "in";
      node = node.parentElement;
    }

    return "in";
  }

  function extractContent(el) {
    const contentEl = el.querySelector(
      ".markdown, .prose, .response-content, [class*='message-text']"
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
        topic: extractNotebookTitle(),
      });
    }
  }

  function init() {
    // Prefer observing the chat log container if present
    const chatLog = document.querySelector("[role='log']");
    UCW.observeChat(chatLog || document.body, processMessages, 500);
    console.log("[UCW] NotebookLM interceptor active");
  }

  if (document.readyState === "complete") init();
  else window.addEventListener("load", init);
})();
