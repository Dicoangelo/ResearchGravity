/**
 * UCW Sovereign Capture — YouTube Watch Tracker
 *
 * Tracks YouTube video watches and captures metadata.
 * Optionally fetches auto-generated transcripts when available.
 */

(function () {
  "use strict";

  const PLATFORM = "youtube";
  let lastVideoId = null;
  let watchStartTime = null;

  function getVideoId() {
    const params = new URLSearchParams(window.location.search);
    return params.get("v");
  }

  function getVideoMetadata() {
    const title = document.querySelector(
      "h1.ytd-video-primary-info-renderer, h1.ytd-watch-metadata yt-formatted-string"
    )?.innerText || document.title;

    const channel = document.querySelector(
      "#channel-name a, ytd-channel-name a"
    )?.innerText || "";

    const description = document.querySelector(
      "#description-inline-expander, #description"
    )?.innerText?.substring(0, 500) || "";

    return { title, channel, description };
  }

  function captureVideoWatch() {
    const videoId = getVideoId();
    if (!videoId || videoId === lastVideoId) return;

    lastVideoId = videoId;
    watchStartTime = Date.now();

    // Wait for metadata to load
    setTimeout(() => {
      const meta = getVideoMetadata();
      const content = `Watching: "${meta.title}" by ${meta.channel}\n${meta.description}`;

      chrome.runtime.sendMessage({
        type: "UCW_CAPTURE",
        platform: PLATFORM,
        event: {
          platform: PLATFORM,
          content,
          direction: "in",
          url: window.location.href,
          topic: meta.title,
          metadata: {
            video_id: videoId,
            channel: meta.channel,
            watch_start: new Date().toISOString(),
          },
        },
      });

      console.log(`[UCW] YouTube watch captured: ${meta.title}`);
    }, 3000); // Wait 3s for metadata to populate
  }

  // Watch for navigation (YouTube is SPA)
  let lastUrl = window.location.href;
  const urlObserver = new MutationObserver(() => {
    if (window.location.href !== lastUrl) {
      lastUrl = window.location.href;
      if (window.location.pathname === "/watch") {
        captureVideoWatch();
      }
    }
  });

  function init() {
    if (window.location.pathname === "/watch") {
      captureVideoWatch();
    }
    urlObserver.observe(document.body, { childList: true, subtree: true });
    console.log("[UCW] YouTube tracker active");
  }

  if (document.readyState === "complete") init();
  else window.addEventListener("load", init);
})();
