/**
 * UCW Sovereign Capture — Grok Interceptor
 *
 * Captures messages from grok.x.ai by observing DOM mutations.
 */

(function () {
  "use strict";

  const PLATFORM = "grok";
  const seenMessages = new WeakSet();

  function processMessages() {
    // Grok uses message containers with role indicators
    const messages = document.querySelectorAll(
      "[class*='message'], [data-testid*='message'], .prose"
    );

    for (const msg of messages) {
      if (seenMessages.has(msg)) continue;
      seenMessages.add(msg);

      const content = msg.innerText?.trim();
      if (!content || content.length < 5) continue;

      // Detect direction from parent/sibling elements
      const parent = msg.closest("[class*='user'], [class*='assistant'], [class*='human'], [class*='ai']");
      const parentClass = (parent?.className || msg.className || "").toLowerCase();
      const direction = parentClass.includes("user") || parentClass.includes("human") ? "out" : "in";

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
    console.log("[UCW] Grok interceptor active");
  }

  if (document.readyState === "complete") init();
  else window.addEventListener("load", init);
})();
