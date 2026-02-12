# UCW Sovereign Capture Chrome Extension

**Status:** Draft | **Created:** 2026-02-11 | **Priority:** High

## Problem

UCW currently captures Claude (CLI, Code, Desktop, CCC) in real-time via local adapters. But ChatGPT (60K events), Grok (20K), and YouTube/NotebookLM data are one-time imports only. No live connection exists for external platforms.

## Solution

A single Chrome extension that intercepts AI conversations and media consumption across all platforms in real-time, routing everything to the local UCW capture endpoint at `localhost:3847`.

## Target Platforms

### AI Conversations (Layer 1 - Core)

| Platform | Method | Data Captured |
|----------|--------|---------------|
| **ChatGPT** | `fetch()` override on `chat.openai.com` | Prompts, responses, model, timestamps |
| **Grok** | `fetch()` override on `grok.x.ai` | Prompts, responses, model, timestamps |
| **Gemini** | `fetch()` override on `gemini.google.com` | Prompts, responses, model, timestamps |
| **Claude Web** | `fetch()` override on `claude.ai` | Prompts, responses, model, timestamps |

### Media & Knowledge (Layer 2 - Enrichment)

| Platform | Method | Data Captured |
|----------|--------|---------------|
| **YouTube** | Page observer + video events | Video ID, title, channel, watch duration, % watched |
| **YouTube Transcripts** | Auto-fetch via `youtube-transcript-api` | Full transcript text, timestamps per segment |
| **NotebookLM** | `fetch()` override on `notebooklm.google.com` | Notebook ID, sources, generated content, audio overviews |

## Architecture

```
Browser (Chrome Extension)
  ├── content_scripts/
  │   ├── ai-interceptor.js      # fetch/XHR override for AI platforms
  │   ├── youtube-tracker.js     # Video watch events + page observer
  │   └── notebooklm-tracker.js  # NotebookLM activity capture
  ├── background/
  │   └── service-worker.js      # Queue, batch, POST to localhost
  ├── popup/
  │   └── popup.html             # Status dashboard, toggle platforms
  └── manifest.json              # MV3, permissions, content script matches

  ──POST──> localhost:3847/api/v2/capture/extension
            (UCW API server, already running)
```

### Data Flow

1. Content script intercepts `fetch()`/`XMLHttpRequest` on target domains
2. Filters for AI API calls (ignores static assets, analytics, etc.)
3. Extracts prompt, response, model, timestamp, conversation ID
4. Sends to service worker via `chrome.runtime.sendMessage()`
5. Service worker batches events (5s window or 10 events, whichever first)
6. POST batch to `localhost:3847/api/v2/capture/extension`
7. UCW server processes: stores event, generates embedding, checks coherence

### UCW Integration

New API endpoint needed on the UCW server:

```
POST /api/v2/capture/extension
Content-Type: application/json

{
  "events": [
    {
      "platform": "chatgpt",
      "event_type": "message",
      "role": "user|assistant",
      "content": "...",
      "model": "gpt-4o",
      "conversation_id": "...",
      "timestamp": "2026-02-11T20:00:00Z",
      "metadata": { "url": "...", "source": "extension" }
    }
  ]
}
```

## Platform-Specific Capture Details

### ChatGPT (`chat.openai.com`)

**Intercept target:** `fetch()` calls to `https://chatgpt.com/backend-api/conversation`

```javascript
// Streaming responses use EventSource/ReadableStream
// Capture the request body (user message) and accumulate streamed chunks
const originalFetch = window.fetch;
window.fetch = async function(...args) {
  const [url, options] = args;
  if (url.includes('/backend-api/conversation')) {
    const body = JSON.parse(options.body);
    captureUserMessage(body);
    const response = await originalFetch.apply(this, args);
    // Clone and read the stream for assistant response
    return interceptStream(response);
  }
  return originalFetch.apply(this, args);
};
```

### Grok (`grok.x.ai`)

**Intercept target:** `fetch()` calls to Grok's conversation API
- Similar pattern to ChatGPT
- May use different streaming format

### Gemini (`gemini.google.com`)

**Intercept target:** `fetch()` calls to `generativelanguage.googleapis.com` or internal Gemini endpoints
- Google uses batched RPC format — need to parse Protobuf-like responses
- Interactions API (beta) may simplify this

### YouTube (`youtube.com`)

**No fetch override needed.** Use DOM observation + video element events:

```javascript
// Watch for navigation to video pages
const observer = new MutationObserver(() => {
  const video = document.querySelector('video');
  if (video && !video.dataset.ucwTracked) {
    video.dataset.ucwTracked = 'true';
    trackVideo(video);
  }
});

function trackVideo(video) {
  const videoId = new URLSearchParams(window.location.search).get('v');
  const startTime = Date.now();

  video.addEventListener('ended', () => {
    sendWatchEvent(videoId, video.duration, video.currentTime);
  });

  // Also capture on navigate-away (SPA navigation)
  window.addEventListener('yt-navigate-start', () => {
    sendWatchEvent(videoId, video.duration, video.currentTime);
  });
}
```

**Transcript auto-fetch:** After capturing a watch event, the service worker calls the local UCW server which fetches the transcript via `youtube-transcript-api` server-side.

### NotebookLM (`notebooklm.google.com`)

**Intercept target:** `fetch()` calls to NotebookLM's internal API
- Capture: notebook opens, source additions, AI-generated content requests
- Audio overview generation events
- Study guide / FAQ generation

## Popup UI

Minimal dark-themed popup showing:

```
UCW Capture Extension          [ON/OFF]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ChatGPT    ● Connected    47 today
  Grok       ● Connected    12 today
  Gemini     ● Connected     3 today
  YouTube    ● Connected    28 today
  NotebookLM ● Connected     2 today
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Server: localhost:3847 ● Online
  Queue: 0 pending | Last sync: 2s ago
```

## Privacy & Sovereignty

- **All data stays local** — POST to localhost only, never to external servers
- **No analytics, no tracking, no telemetry**
- **User controls which platforms are active** via popup toggles
- **Content filtering** — option to exclude sensitive conversations
- **Open source** — user owns the code

## Schema Additions

New tables needed in UCW PostgreSQL:

```sql
-- YouTube watch history
CREATE TABLE youtube_watch_history (
  id SERIAL PRIMARY KEY,
  video_id VARCHAR(20) NOT NULL,
  title TEXT,
  channel_name TEXT,
  channel_id VARCHAR(50),
  watched_at TIMESTAMPTZ NOT NULL,
  watch_duration_s INTEGER,
  watch_percentage REAL,
  source VARCHAR(20) DEFAULT 'extension',
  metadata JSONB,
  UNIQUE(video_id, watched_at)
);

-- YouTube transcripts
CREATE TABLE youtube_transcripts (
  id SERIAL PRIMARY KEY,
  video_id VARCHAR(20) NOT NULL UNIQUE,
  language VARCHAR(10) DEFAULT 'en',
  transcript JSONB,
  full_text TEXT,
  embedding vector(768),
  captured_at TIMESTAMPTZ DEFAULT NOW()
);

-- NotebookLM notebooks
CREATE TABLE notebooklm_notebooks (
  id SERIAL PRIMARY KEY,
  notebook_id VARCHAR(255) UNIQUE,
  title TEXT,
  sources JSONB,
  metadata JSONB,
  created_at TIMESTAMPTZ,
  last_synced TIMESTAMPTZ DEFAULT NOW()
);

-- NotebookLM artifacts (audio overviews, study guides, etc.)
CREATE TABLE notebooklm_artifacts (
  id SERIAL PRIMARY KEY,
  notebook_id VARCHAR(255) REFERENCES notebooklm_notebooks(notebook_id),
  artifact_type VARCHAR(50),
  file_path TEXT,
  content_text TEXT,
  embedding vector(768),
  generated_at TIMESTAMPTZ,
  captured_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Implementation Phases

### Phase 1: Core AI Capture (3-4 days)

- [ ] Chrome extension scaffold (MV3, service worker, popup)
- [ ] `fetch()` interceptor for ChatGPT
- [ ] UCW server endpoint: `POST /api/v2/capture/extension`
- [ ] Store intercepted events in `cognitive_events` table
- [ ] Test: capture a full ChatGPT conversation

### Phase 2: Multi-Platform AI (2-3 days)

- [ ] Add Grok interceptor
- [ ] Add Gemini interceptor (handle Google's RPC format)
- [ ] Add Claude web interceptor
- [ ] Popup UI with per-platform toggles
- [ ] Batch queue in service worker

### Phase 3: YouTube Capture (2-3 days)

- [ ] YouTube watch tracker (video events + SPA navigation)
- [ ] Schema migration for `youtube_watch_history` and `youtube_transcripts`
- [ ] Server-side transcript fetch via `youtube-transcript-api`
- [ ] Auto-embed transcripts for coherence detection
- [ ] Google Takeout importer for historical backfill

### Phase 4: NotebookLM Capture (2 days)

- [ ] NotebookLM interceptor
- [ ] Schema migration for `notebooklm_notebooks` and `notebooklm_artifacts`
- [ ] Audio overview detection and download
- [ ] Daily backup via `notebooklm-py` (optional, for full artifacts)

### Phase 5: Coherence Integration (1-2 days)

- [ ] Extension events flow through coherence engine
- [ ] YouTube transcripts included in cross-platform coherence detection
- [ ] NotebookLM insights detected as coherence moments
- [ ] Dashboard updated to show YouTube + NotebookLM platforms
- [ ] FSRS scheduling for YouTube/NotebookLM insights

## Dependencies

| Dependency | Purpose | Status |
|-----------|---------|--------|
| UCW API server (`:3847`) | Capture endpoint | Already running |
| PostgreSQL + pgvector | Storage | Already running |
| `youtube-transcript-api` | Transcript fetch | `pip install` needed |
| `notebooklm-py` | Full artifact backup | `pip install` needed |
| Chrome MV3 | Extension platform | Available |

## Success Metrics

| Metric | Target |
|--------|--------|
| ChatGPT capture latency | < 1s from message to database |
| YouTube transcript coverage | > 90% of watched videos |
| Platform uptime | Extension survives page refreshes, SPA navigation |
| Zero data loss | Queue + retry ensures no dropped events |
| Coherence detection | YouTube/NotebookLM moments appearing in dashboard |

## Open Questions

1. **Gemini's RPC format** — need to reverse-engineer the protobuf structure
2. **NotebookLM audio** — can we capture the generated audio blob directly, or only metadata?
3. **YouTube shorts** — different URL pattern (`/shorts/ID`), need separate tracker?
4. **Rate limiting** — how fast can we POST to the local server without impacting browser performance?
5. **Extension store** — publish to Chrome Web Store or sideload only?

## References

- Urban VPN incident (Dec 2025): Proof that fetch override captures across 10+ AI platforms
- OpenAI Conversations API: `platform.openai.com/docs/api-reference/conversations`
- xAI Responses API: `docs.x.ai/docs/guides/responses-api`
- Gemini Interactions API (beta): `ai.google.dev/gemini-api/docs/interactions`
- `youtube-transcript-api`: `github.com/jdepoix/youtube-transcript-api`
- `notebooklm-py`: `github.com/teng-lin/notebooklm-py`
- Chrome MV3 docs: `developer.chrome.com/docs/extensions/mv3`
