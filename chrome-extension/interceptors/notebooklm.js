/**
 * UCW Sovereign Capture — NotebookLM Interceptor
 *
 * Captures interactions from notebooklm.google.com.
 */

(function () {
  "use strict";

  const PLATFORM = "notebooklm";
  const seenMessages = new WeakSet();

  function processMessages() {
    // NotebookLM uses chat-like containers for Q&A
    const messages = document.querySelectorAll(
      "[class*='message'], [class*='response'], [class*='query'], .chat-message"
    );

    for (const msg of messages) {
      if (seenMessages.has(msg)) continue;
      seenMessages.add(msg);

      const content = msg.innerText?.trim();
      if (!content || content.length < 5) continue;

      const className = (msg.className || "").toLowerCase();
      const direction = className.includes("query") || className.includes("user") ? "out" : "in";

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
    console.log("[UCW] NotebookLM interceptor active");
  }

  if (document.readyState === "complete") init();
  else window.addEventListener("load", init);
})();
