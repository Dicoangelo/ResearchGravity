/**
 * UCW Sovereign Capture — ChatGPT Interceptor v0.2.0
 *
 * Captures messages from chatgpt.com / chat.openai.com
 * Uses: MutationObserver on [data-message-author-role] elements.
 * Extracts: content, direction, session_id, conversation title, model.
 */

/* global UCW */

(function () {
  "use strict";

  const PLATFORM = "chatgpt";
  const seenMessages = new WeakSet();

  function extractSessionId() {
    const match = window.location.pathname.match(/\/(?:c|chat)\/([a-f0-9-]+)/);
    return match ? `chatgpt-${match[1]}` : null;
  }

  function extractTitle() {
    // ChatGPT shows the conversation title in the nav or header
    const titleEl =
      document.querySelector("nav a.bg-token-sidebar-surface-secondary") ||
      document.querySelector("h1") ||
      document.querySelector("[data-testid='conversation-title']");
    return titleEl?.textContent?.trim() || "";
  }

  function extractModel() {
    // Model selector button often shows current model
    const modelEl = document.querySelector(
      "[data-testid='model-switcher'] span, button[aria-label*='model'] span"
    );
    return modelEl?.textContent?.trim() || "";
  }

  function processMessages() {
    const messages = document.querySelectorAll("[data-message-author-role]");
    for (const msg of messages) {
      if (seenMessages.has(msg)) continue;
      seenMessages.add(msg);

      const role = msg.getAttribute("data-message-author-role");
      if (!role) continue;

      const direction = role === "user" ? "out" : "in";
      const contentEl = msg.querySelector(".markdown, .whitespace-pre-wrap");
      const content = contentEl ? contentEl.innerText : msg.innerText;

      if (!content || content.trim().length < 5) continue;

      UCW.captureEvent(PLATFORM, content, direction, {
        session_hint: extractSessionId(),
        topic: extractTitle(),
        metadata: {
          model: extractModel(),
          role,
        },
      });
    }
  }

  function init() {
    UCW.observeChat(document.body, processMessages, 500);
    console.log("[UCW] ChatGPT interceptor active");
  }

  if (document.readyState === "complete") init();
  else window.addEventListener("load", init);
})();
