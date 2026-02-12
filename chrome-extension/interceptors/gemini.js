/**
 * UCW Sovereign Capture — Gemini Interceptor
 *
 * Captures messages from gemini.google.com by observing DOM mutations.
 */

(function () {
  "use strict";

  const PLATFORM = "gemini";
  const seenMessages = new WeakSet();

  function processMessages() {
    // Gemini renders messages in model-response and user-query containers
    const allMessages = document.querySelectorAll(
      "model-response, user-query, .message-content, [class*='query-text'], [class*='response-text']"
    );

    for (const msg of allMessages) {
      if (seenMessages.has(msg)) continue;
      seenMessages.add(msg);

      const content = msg.innerText?.trim();
      if (!content || content.length < 5) continue;

      const tagName = msg.tagName?.toLowerCase() || "";
      const className = (msg.className || "").toLowerCase();
      const direction =
        tagName === "user-query" || className.includes("query")
          ? "out"
          : "in";

      chrome.runtime.sendMessage({
        type: "UCW_CAPTURE",
        platform: PLATFORM,
        event: {
          platform: PLATFORM,
          content: content.substring(0, 10000),
          direction,
          url: window.location.href,
        },
      });
    }
  }

  const observer = new MutationObserver(() => processMessages());

  function init() {
    processMessages();
    observer.observe(document.body, { childList: true, subtree: true });
    console.log("[UCW] Gemini interceptor active");
  }

  if (document.readyState === "complete") init();
  else window.addEventListener("load", init);
})();
