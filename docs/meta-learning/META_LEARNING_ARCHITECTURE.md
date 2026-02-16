# Meta-Learning Engine: Architecture & Integration

## Design Philosophy

**Standalone-First, Universal Integration**

The Meta-Learning Engine follows the **Antigravity Innovation Pattern**:

1. **Build standalone** - Works independently with clear APIs
2. **Integrate universally** - Any app can consume it

### Antigravity Pattern Examples

This is the same pattern used across the ecosystem:

| Innovation | Standalone Location | Integration Layer | Consumers |
|------------|---------------------|-------------------|-----------|
| **CPB** (Cognitive Precision Bridge) | `~/researchgravity/cpb/` | Python import, CLI | ResearchGravity, meta-vengine |
| **VoiceNexus** | `@metaventionsai/voice-nexus` (npm) | TypeScript SDK | OS-App, any React app |
| **Meta-Learning Engine** | `~/researchgravity/` HTTP API | Python import, TypeScript SDK, CLI | ResearchGravity, OS-App, meta-vengine |

**Philosophy:** Build once standalone, integrate everywhere.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          META-LEARNING ENGINE (Standalone Service)          │
│                  localhost:3847 (FastAPI)                    │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Storage Layer: SQLite + Qdrant (1024d vectors)       │ │
│  │  • 666 session outcomes                                │ │
│  │  • 535 cognitive states                                │ │
│  │  • 30 error patterns                                   │ │
│  │  • 2,530 research findings                             │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Correlation Engine: Multi-dimensional predictions    │ │
│  │  • Temporal joins                                      │ │
│  │  • Multi-vector search                                 │ │
│  │  • Calibration loop                                    │ │
│  │  • Adaptive weights                                    │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  REST API: 7 HTTP/JSON endpoints                      │ │
│  │  POST /api/v2/predict/session                          │ │
│  │  POST /api/v2/predict/errors                           │ │
│  │  POST /api/v2/predict/optimal-time                     │ │
│  │  GET  /api/v2/predict/accuracy                         │ │
│  │  POST /api/v2/predict/update-outcome                   │ │
│  │  GET  /api/v2/predict/multi-search                     │ │
│  │  GET  /api/v2/predict/calibrate-weights                │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
┌───────────────┐   ┌──────────────────┐   ┌──────────────┐
│ ResearchGravity│   │     OS-App       │   │ meta-vengine │
│ (Python CLI)   │   │ (React + TypeScript)│   │ (Python)   │
│                │   │                  │   │              │
│ Direct HTTP    │   │ TypeScript SDK   │   │ Direct HTTP  │
│ or CLI tools   │   │ + React hooks    │   │ or Python    │
└───────────────┘   └──────────────────┘   └──────────────┘
        ↓                     ↓                     ↓
  Session init       Knowledge Injector    Session optimizer
  predictions        UI predictions        Cognitive routing
  Error prevention   Dashboard widgets     Timing suggestions
```

---

## Standalone Operation

### Start the Service

```bash
cd ~/researchgravity
source .venv/bin/activate
python3 -m api.server --port 3847
```

**Requirements:**
- Python 3.8+
- SQLite database (`~/.agent-core/storage/antigravity.db`)
- Qdrant running (localhost:6333)
- Cohere API key

**No dependencies on:**
- OS-App
- meta-vengine
- Any specific consumer

### CLI Usage (No Integration Required)

```bash
# Predict session outcome
python3 predict_session.py "implement authentication" --hour 20 --verbose

# Predict potential errors
python3 predict_errors.py "git clone repository"

# Use HTTP client
python3 predict_api_client.py predict "build feature X" --track

# Check accuracy
python3 predict_api_client.py accuracy --days 30
```

**The engine works completely standalone.**

---

## Integration Options

### Option 1: Direct HTTP (Universal)

Any language/platform can consume via HTTP:

```bash
# cURL
curl -X POST http://localhost:3847/api/v2/predict/session \
  -H "Content-Type: application/json" \
  -d '{"intent": "implement auth", "track_prediction": true}'

# Python requests
import requests
response = requests.post(
    "http://localhost:3847/api/v2/predict/session",
    json={"intent": "implement auth", "track_prediction": True}
)
prediction = response.json()

# JavaScript fetch
const response = await fetch('http://localhost:3847/api/v2/predict/session', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ intent: 'implement auth', track_prediction: true })
});
const prediction = await response.json();
```

### Option 2: Python Integration (ResearchGravity, meta-vengine)

```python
# Direct import
from storage.meta_learning import MetaLearningEngine

engine = MetaLearningEngine()
await engine.initialize()

prediction = await engine.predict_session_outcome(
    intent="implement feature",
    cognitive_state={"mode": "peak", "hour": 20}
)

# Or via HTTP client
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:3847/api/v2/predict/session",
        json={"intent": "implement feature"}
    )
    prediction = response.json()
```

### Option 3: TypeScript SDK (OS-App)

**Optional convenience layer** for TypeScript applications:

```typescript
// Install SDK
import { AgentCoreClient, useSessionPrediction } from '@antigravity/agent-core-sdk';

// Direct client usage
const client = new AgentCoreClient({ baseUrl: 'http://localhost:3847' });
const prediction = await client.predictSession({ intent: 'implement auth' });

// React hook usage
function MyComponent() {
  const { prediction, isLoading } = useSessionPrediction({
    intent: 'implement auth',
    track: true
  });

  return <div>{prediction?.predicted_quality}/5</div>;
}
```

**Note:** The SDK is just a wrapper around the HTTP API. Not required.

---

## Consumer Applications

### 1. ResearchGravity (Primary)

**Use Case:** Session initialization and research planning

```bash
# Before starting a session
python3 predict_session.py "multi-agent orchestration research"
# Output: Quality 4.2/5, Success 78%, Optimal time: 20:00

# Initialize session if prediction is favorable
python3 init_session.py "multi-agent orchestration"
```

**Integration Point:** `init_session.py` can call prediction API before creating session

### 2. OS-App (UI/Frontend)

**Use Case:** Real-time predictions in user interface

```typescript
// Knowledge Injector enhancement
async function injectContext(query: string) {
  const { prediction, errors } = await agentCore.getPredictionWithContext(query);

  if (prediction.success_probability < 0.7) {
    showWarning(`Low success probability: ${prediction.success_probability}`);
  }

  return selectContext(query, prediction.recommended_research);
}

// Dashboard widget
function PredictionPanel({ intent }: { intent: string }) {
  const { prediction } = useSessionPrediction({ intent });
  return <QualityBadge quality={prediction?.predicted_quality} />;
}
```

**Integration Point:** TypeScript SDK provides React hooks for UI components

### 3. meta-vengine (Session Optimizer)

**Use Case:** Cognitive routing and session scheduling

```python
# Session optimizer
async def should_schedule_later(intent: str) -> bool:
    prediction = await get_prediction(intent)

    if prediction["success_probability"] < 0.6:
        optimal_time = prediction["optimal_time"]
        print(f"Schedule for {optimal_time}:00 instead")
        return True

    return False

# Cognitive OS integration
from storage.meta_learning import MetaLearningEngine

engine = MetaLearningEngine()
await engine.initialize()

# Get prediction based on current cognitive state
current_state = get_cognitive_state()  # from Cognitive OS
prediction = await engine.predict_session_outcome(
    intent=task_intent,
    cognitive_state=current_state
)
```

**Integration Point:** Direct Python import or HTTP API

### 4. Future Applications

Any tool in the ecosystem can integrate:

```bash
# Custom CLI tool
#!/bin/bash
INTENT="$1"
curl -X POST http://localhost:3847/api/v2/predict/session \
  -H "Content-Type: application/json" \
  -d "{\"intent\": \"$INTENT\"}" | jq '.predicted_quality'

# Custom Python script
import httpx

async def check_if_ready(task: str) -> bool:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:3847/api/v2/predict/session",
            json={"intent": task}
        )
        prediction = resp.json()
        return prediction["success_probability"] > 0.75
```

---

## Integration Layers

### Layer 1: Core Service (Standalone)

- Python backend
- SQLite + Qdrant storage
- REST API server
- CLI tools

**Status:** ✅ Complete (Phases 1-5)
**Dependencies:** None (runs standalone)

### Layer 2: Python Consumers (ResearchGravity, meta-vengine)

- Direct import of `MetaLearningEngine`
- HTTP client via `httpx` or `requests`
- CLI tool invocation

**Status:** ✅ Available (built-in)
**Dependencies:** Meta-Learning Engine running

### Layer 3: TypeScript SDK (OS-App)

- Type-safe client
- React hooks
- Convenience wrappers

**Status:** ✅ Complete (Phase 6)
**Dependencies:** Meta-Learning Engine running
**Note:** Optional - OS-App can use HTTP directly

### Layer 4: UI Components (Future)

- React components for predictions
- Dashboard widgets
- Prediction visualizations

**Status:** 🔜 Planned (Phase 7)
**Dependencies:** TypeScript SDK

---

## Data Flow

### 1. Training Data Collection

```
User Sessions → ResearchGravity → SQLite + Qdrant
     ↓                               ↓
Cognitive OS → States → SQLite + Qdrant
     ↓                               ↓
Error Recovery → Patterns → SQLite + Qdrant
```

### 2. Prediction Generation

```
Application → HTTP Request → Meta-Learning Engine
                    ↓
              Correlation Engine
                    ↓
           Multi-vector Search
                    ↓
            Weighted Scoring
                    ↓
           HTTP Response → Application
```

### 3. Calibration Loop

```
Application → Track Prediction → Store in DB
     ↓
Execute Session
     ↓
Update Outcome → Compare → Adjust Weights
     ↓
Next Prediction (improved)
```

---

## Deployment Modes

### Development

```bash
# Terminal 1: Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Terminal 2: Start Meta-Learning API
cd ~/researchgravity
source .venv/bin/activate
python3 -m api.server --port 3847

# Terminal 3: Use from any app
python3 predict_session.py "task"
# or
cd ~/OS-App && npm run dev
```

### Production (Future)

```bash
# Background service
systemctl start meta-learning-engine

# Or Docker
docker run -d -p 3847:3847 antigravity/meta-learning-engine
```

---

## Key Design Decisions

### Why Standalone Service?

1. **Separation of Concerns**: Predictions don't depend on any specific app
2. **Reusability**: Any tool can consume predictions
3. **Independent Scaling**: Can optimize/cache predictions separately
4. **Technology Agnostic**: HTTP API works with any language
5. **Testability**: Can test engine without UI/consumers

### Why HTTP API?

1. **Language-Independent**: Python, TypeScript, bash, etc.
2. **Network-Ready**: Can run on different machines
3. **Standard Protocol**: Well-understood, easy to debug
4. **Client Flexibility**: Direct HTTP, SDK, CLI - all options available

### Why Optional SDKs?

1. **Convenience**: TypeScript SDK makes React integration easier
2. **Type Safety**: Interfaces prevent API misuse
3. **Not Required**: Apps can use HTTP directly if preferred
4. **Consumer Choice**: Pick the integration method that fits best

---

## Summary

**The Meta-Learning Engine is:**
- ✅ Standalone HTTP service (localhost:3847)
- ✅ Consumable by any application
- ✅ Independent of OS-App, ResearchGravity, or meta-vengine
- ✅ Integrated via HTTP API, Python import, or TypeScript SDK

**Consumers:**
- ResearchGravity: Session predictions, CLI tools
- OS-App: UI predictions, React hooks (optional SDK)
- meta-vengine: Session optimization, cognitive routing
- Any future tool: Direct HTTP integration

**The TypeScript SDK (Phase 6) is an optional convenience layer, not a requirement.**

---

**Architecture:** Microservice (standalone-first)
**Integration:** Universal (HTTP, Python, TypeScript)
**Status:** Production-ready for all consumers 🚀
