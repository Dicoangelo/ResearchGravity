/**
 * UCW Sovereign Capture — ChatGPT Interceptor
 *
 * Captures messages from chatgpt.com by observing DOM mutations
 * in the chat container.
 */

(function () {
  "use strict";

  const PLATFORM = "chatgpt";
  const seenMessages = new WeakSet();

  function extractMessage(el) {
    // ChatGPT uses data-message-author-role attribute
    const role = el.getAttribute("data-message-author-role");
    if (!role) return null;

    const direction = role === "user" ? "out" : "in";
    const contentEl = el.querySelector(".markdown, .whitespace-pre-wrap");
    const content = contentEl ? contentEl.innerText : el.innerText;

    if (!content || content.trim().length < 5) return null;

    return { content: content.substring(0, 10000), direction };
  }

  function processMessages() {
    // ChatGPT renders messages in article or div[data-message-author-role] elements
    const messages = document.querySelectorAll("[data-message-author-role]");
    for (const msg of messages) {
      if (seenMessages.has(msg)) continue;
      seenMessages.add(msg);

      const extracted = extractMessage(msg);
      if (extracted) {
        chrome.runtime.sendMessage({
          type: "UCW_CAPTURE",
          platform: PLATFORM,
          event: {
            platform: PLATFORM,
            content: extracted.content,
            direction: extracted.direction,
            url: window.location.href,
            session_hint: extractSessionId(),
          },
        });
      }
    }
  }

  function extractSessionId() {
    // ChatGPT URL format: /c/uuid or /chat/uuid
    const match = window.location.pathname.match(/\/(?:c|chat)\/([a-f0-9-]+)/);
    return match ? `chatgpt-${match[1]}` : null;
  }

  // Observe for new messages
  const observer = new MutationObserver(() => {
    processMessages();
  });

  function init() {
    processMessages();
    observer.observe(document.body, { childList: true, subtree: true });
    console.log("[UCW] ChatGPT interceptor active");
  }

  // Wait for page load
  if (document.readyState === "complete") {
    init();
  } else {
    window.addEventListener("load", init);
  }
})();
