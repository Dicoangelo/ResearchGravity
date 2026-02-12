/**
 * UCW Sovereign Capture — YouTube Watch Tracker v0.2.0
 *
 * Tracks YouTube video watches and captures metadata.
 * YouTube is an SPA — uses MutationObserver to detect navigation.
 */

/* global UCW */

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
    const title =
      document.querySelector(
        "h1.ytd-video-primary-info-renderer, h1.ytd-watch-metadata yt-formatted-string"
      )?.innerText || document.title;

    const channel =
      document.querySelector("#channel-name a, ytd-channel-name a")
        ?.innerText || "";

    const description =
      document
        .querySelector("#description-inline-expander, #description")
        ?.innerText?.substring(0, 500) || "";

    return { title, channel, description };
  }

  function captureVideoWatch() {
    const videoId = getVideoId();
    if (!videoId || videoId === lastVideoId) return;

    // Capture watch duration for previous video
    if (lastVideoId && watchStartTime) {
      const durationSec = Math.round((Date.now() - watchStartTime) / 1000);
      if (durationSec > 10) {
        UCW.captureEvent(PLATFORM, `Watched for ${durationSec}s`, "in", {
          metadata: {
            video_id: lastVideoId,
            watch_duration_sec: durationSec,
            event_type: "watch_end",
          },
        });
      }
    }

    lastVideoId = videoId;
    watchStartTime = Date.now();

    // Wait for metadata to load
    setTimeout(() => {
      const meta = getVideoMetadata();
      const content = `Watching: "${meta.title}" by ${meta.channel}\n${meta.description}`;

      UCW.captureEvent(PLATFORM, content, "in", {
        topic: meta.title,
        metadata: {
          video_id: videoId,
          channel: meta.channel,
          watch_start: new Date().toISOString(),
          event_type: "watch_start",
        },
      });

      console.log(`[UCW] YouTube watch captured: ${meta.title}`);
    }, 3000);
  }

  function init() {
    if (window.location.pathname === "/watch") {
      captureVideoWatch();
    }

    // Use proper SPA navigation observer (hooks pushState/replaceState/popstate)
    UCW.observeUrl((newUrl) => {
      const url = new URL(newUrl);
      if (url.pathname === "/watch") {
        captureVideoWatch();
      }
    }, 500);

    console.log("[UCW] YouTube tracker active");
  }

  if (document.readyState === "complete") init();
  else window.addEventListener("load", init);
})();
