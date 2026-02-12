/**
 * UCW Sovereign Capture — Shared capture utilities v0.3.0
 *
 * Loaded before each platform interceptor via manifest content_scripts.
 * Provides: captureEvent(), observeChat(), observeUrl().
 */

/* global chrome */

const UCW = (() => {
  "use strict";

  const MIN_CONTENT_LENGTH = 5;
  const MAX_CONTENT_LENGTH = 10000;

  /**
   * Send a captured event to the background service worker.
   */
  function captureEvent(platform, content, direction, opts = {}) {
    if (!content || content.trim().length < MIN_CONTENT_LENGTH) return;

    const event = {
      platform,
      content: content.substring(0, MAX_CONTENT_LENGTH),
      direction,
      url: opts.url || window.location.href,
      topic: opts.topic || "",
      intent: opts.intent || "",
      concepts: opts.concepts || [],
      session_hint: opts.session_hint || null,
      metadata: opts.metadata || {},
    };

    try {
      chrome.runtime.sendMessage(
        { type: "UCW_CAPTURE", platform, event },
        (response) => {
          // Handle "Extension context invalidated" gracefully
          if (chrome.runtime.lastError) {
            // Silently ignore — extension was reloaded or disabled
            return;
          }
          if (response?.status === "queued") {
            console.debug(
              `[UCW] Captured ${direction} on ${platform} (queue: ${response.queueSize})`
            );
          }
        }
      );
    } catch (err) {
      // Extension context invalidated — stop trying
      console.debug("[UCW] Extension context lost, stopping capture");
    }
  }

  /**
   * Debounced MutationObserver wrapper.
   * Calls processFunc at most once per `delayMs` when mutations occur.
   */
  function observeChat(target, processFunc, delayMs = 300) {
    let timer = null;
    let alive = true;

    function debouncedProcess() {
      if (!alive) return;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        if (alive) processFunc();
      }, delayMs);
    }

    // Initial scan
    processFunc();

    const el = typeof target === "string"
      ? document.querySelector(target) || document.body
      : target || document.body;

    const observer = new MutationObserver(debouncedProcess);
    observer.observe(el, { childList: true, subtree: true });

    // Return cleanup function
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
      observer.disconnect();
    };
  }

  /**
   * Debounced URL-change observer for SPA platforms (YouTube, etc.).
   * Calls onChange(newUrl) when window.location.href changes.
   */
  function observeUrl(onChange, delayMs = 200) {
    let lastUrl = window.location.href;
    let timer = null;
    let alive = true;

    function check() {
      if (!alive) return;
      const current = window.location.href;
      if (current !== lastUrl) {
        lastUrl = current;
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => {
          if (alive) onChange(current);
        }, delayMs);
      }
    }

    const observer = new MutationObserver(check);
    observer.observe(document.body, { childList: true, subtree: true });

    // Also listen for History API navigation
    const origPush = history.pushState;
    const origReplace = history.replaceState;
    history.pushState = function () {
      origPush.apply(this, arguments);
      check();
    };
    history.replaceState = function () {
      origReplace.apply(this, arguments);
      check();
    };
    window.addEventListener("popstate", check);

    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
      observer.disconnect();
      history.pushState = origPush;
      history.replaceState = origReplace;
      window.removeEventListener("popstate", check);
    };
  }

  return { captureEvent, observeChat, observeUrl };
})();
