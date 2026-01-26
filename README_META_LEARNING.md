# Meta-Learning Engine

**Standalone predictive service that learns which [research + cognitive state + tools] combinations lead to successful outcomes.**

---

## TL;DR

```bash
# Start the service (standalone)
python3 -m api.server --port 3847

# Use from Python
from storage.meta_learning import MetaLearningEngine
prediction = await engine.predict_session_outcome(intent="task")

# Use from TypeScript
import { useSessionPrediction } from '@antigravity/agent-core-sdk';
const { prediction } = useSessionPrediction({ intent: 'task' });

# Use from CLI
python3 predict_session.py "implement feature" --verbose

# Use from HTTP
curl http://localhost:3847/api/v2/predict/session -X POST -d '...'
```

---

## Architecture

**Standalone-First, Universal Integration**

The Meta-Learning Engine is a **microservice** built following the Antigravity Innovation Pattern (same as CPB and VoiceNexus):

```
┌─────────────────────────────────────┐
│   Standalone Service (Core)         │  ← Works independently
│   HTTP API @ localhost:3847          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Integration Layers                 │  ← Optional but PLANNED
│   • Python import/CLI                │  ✅ Complete
│   • TypeScript SDK                   │  ✅ Complete
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   UI Components (OS-App)             │  ← Optional but PLANNED
│   • React components                 │  🔜 Phase 7
└─────────────────────────────────────┘
```

**Clarification:**
- **Architecturally Optional** - Service works standalone without SDK/UI
- **Implementation Confirmed** - We're building all layers (not "maybe")

**Why This Matters:**
- Service remains usable without SDK/UI (architectural independence)
- But we're definitely building SDK/UI for convenience (confirmed scope)

---

## Status

| Phase | Component | Status | Note |
|-------|-----------|--------|------|
| 1 | Session Outcomes | ✅ Complete | Standalone core |
| 2 | Cognitive States | ✅ Complete | Standalone core |
| 3 | Error Patterns | ✅ Complete | Standalone core |
| 4 | Correlation Engine | ✅ Complete | Standalone core |
| 5 | REST API | ✅ Complete | Standalone core |
| 6 | TypeScript SDK | ✅ Complete | Optional but **confirmed** |
| 7 | UI Components | 🔜 Planned | Optional but **confirmed** |

**Current:** Phases 1-6 complete (100%), Phase 7 ready to implement
**Architecture:** Standalone service working independently
**Integration:** SDK and UI layers confirmed for implementation

---

## Quick Start

### 1. Start the Service

```bash
cd ~/researchgravity
source .venv/bin/activate
python3 -m api.server --port 3847
```

Service runs at `http://localhost:3847`

### 2. Use from Any Application

**Python (ResearchGravity, meta-vengine):**
```python
# Option 1: Direct import
from storage.meta_learning import MetaLearningEngine
engine = MetaLearningEngine()
await engine.initialize()
prediction = await engine.predict_session_outcome(intent="task")

# Option 2: HTTP client
import httpx
async with httpx.AsyncClient() as client:
    resp = await client.post("http://localhost:3847/api/v2/predict/session",
                             json={"intent": "task"})
    prediction = resp.json()
```

**TypeScript (OS-App):**
```typescript
// Option 1: SDK (convenient, recommended)
import { AgentCoreClient } from '@antigravity/agent-core-sdk';
const client = new AgentCoreClient();
const prediction = await client.predictSession({ intent: 'task' });

// Option 2: Direct HTTP (no SDK required)
const response = await fetch('http://localhost:3847/api/v2/predict/session', {
  method: 'POST',
  body: JSON.stringify({ intent: 'task' })
});
const prediction = await response.json();
```

**CLI:**
```bash
python3 predict_session.py "implement authentication" --verbose
python3 predict_errors.py "git operations"
python3 predict_api_client.py accuracy --days 30
```

**cURL:**
```bash
curl -X POST http://localhost:3847/api/v2/predict/session \
  -H "Content-Type: application/json" \
  -d '{"intent": "implement auth"}' | jq
```

---

## Integration Status

### Confirmed Integrations

**ResearchGravity:**
- ✅ CLI tools (`predict_session.py`, `predict_errors.py`)
- ✅ Python import (`MetaLearningEngine` class)
- 🔜 Integration into `init_session.py` (planned)

**OS-App:**
- ✅ TypeScript SDK (`@antigravity/agent-core-sdk`)
- ✅ React hooks (`useSessionPrediction`, etc.)
- 🔜 UI components (Phase 7 - planned)
- 🔜 Knowledge Injector enhancement (planned)
- 🔜 Dashboard integration (planned)

**meta-vengine:**
- ✅ Python import option
- ✅ HTTP client option
- 🔜 Session optimizer integration (planned)

### Integration Approach

**Standalone First:** Service works independently ✅
**SDK Layer:** TypeScript SDK for convenience ✅ (optional but confirmed)
**UI Layer:** React components for visualization 🔜 (optional but confirmed)

**Note:** "Optional" means architecturally independent, not "maybe we'll build it". All layers are confirmed for implementation.

---

## Consumers

| Application | Method | Status | Notes |
|-------------|--------|--------|-------|
| ResearchGravity | CLI + Python import | ✅ Ready | Session predictions, error prevention |
| OS-App | TypeScript SDK + UI | ✅ SDK ready, 🔜 UI planned | Dashboard widgets, Knowledge Injector |
| meta-vengine | Python import / HTTP | ✅ Ready | Session optimization, cognitive routing |
| Custom scripts | HTTP API | ✅ Ready | Any language, any tool |

---

## Documentation

### Architecture & Design
- `META_LEARNING_ARCHITECTURE.md` - System architecture, integration patterns
- `META_LEARNING_IMPLEMENTATION_COMPLETE.md` - Complete implementation guide

### Quick References
- `META_LEARNING_QUICK_START.md` - Quick start for all consumers
- `README_META_LEARNING.md` - This file (overview)

### Phase Documentation
- `PHASE_1_COMPLETE.md` - Session outcomes (666 vectors)
- `PHASE_2_COMPLETE.md` - Cognitive states (535 vectors)
- `PHASE_3_COMPLETE.md` - Error patterns (30 patterns, 87% prevention)
- `PHASE_4_COMPLETE.md` - Correlation engine (temporal joins, calibration)
- `PHASE_5_COMPLETE.md` - REST API (7 endpoints)
- `PHASE_6_COMPLETE.md` - TypeScript SDK (optional but confirmed)
- `PHASE_7_PLAN.md` - UI components (optional but confirmed)

---

## API Endpoints

Base URL: `http://localhost:3847/api/v2/predict/`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/session` | POST | Predict session outcome (quality, success, errors) |
| `/errors` | POST | Predict potential errors with solutions |
| `/optimal-time` | POST | Find best time for task |
| `/accuracy` | GET | Get prediction accuracy metrics |
| `/update-outcome` | POST | Update prediction with actual outcome |
| `/multi-search` | GET | Multi-dimensional semantic search |
| `/calibrate-weights` | GET | Get weight calibration recommendations |

---

## Key Features

### 1. Multi-Dimensional Predictions
Correlates 4 data dimensions:
- Historical outcomes (666 sessions)
- Cognitive states (535 temporal patterns)
- Research availability (2,530 findings)
- Error patterns (30 preventable errors)

### 2. Predictive Error Prevention
87% average prevention rate:
- Git errors: 95% preventable
- Concurrency issues: 95% preventable
- Permissions: 90% preventable

### 3. Temporal-Cognitive Correlation
Links time-of-day patterns to success:
- Peak hours: 2, 12, 20
- Cognitive mode tracking
- Energy level analysis

### 4. Calibration Loop
Self-improving predictions:
- Track predictions vs actuals
- Adjust correlation weights
- Improve accuracy over time

### 5. Universal Integration
Works with any language/platform:
- Python: Direct import or HTTP
- TypeScript: SDK or HTTP
- Bash: CLI tools or cURL
- Any language: HTTP API

---

## Metrics

**Data:** 1,231 vectors (666 outcomes + 535 cognitive + 30 errors)
**Confidence:** 64% (up from 24% baseline)
**Success Rate:** 77% at optimal times (up from 65%)
**Error Prevention:** 87% average
**API Latency:** <500ms per prediction

---

## Roadmap

### ✅ Complete (Phases 1-6)

- [x] Session outcome vectorization
- [x] Cognitive state vectorization
- [x] Error pattern vectorization
- [x] Correlation engine with calibration
- [x] REST API (7 endpoints)
- [x] TypeScript SDK + React hooks

### 🔜 In Progress (Phase 7)

- [ ] PredictionBadge component
- [ ] ErrorWarningPanel component
- [ ] OptimalTimeIndicator component
- [ ] ResearchChips component
- [ ] PredictionPanel (composite)
- [ ] SignalBreakdown (advanced)
- [ ] Dashboard integration
- [ ] Knowledge Injector enhancement

### 🔮 Future (Phase 8+)

- [ ] Prediction history viewer
- [ ] Calibration dashboard
- [ ] Prediction notifications
- [ ] Automated session scheduling
- [ ] Cross-project pattern transfer

---

## Pattern Recognition

The Meta-Learning Engine follows the **Antigravity Innovation Pattern** - the same architecture as:

**CPB (Cognitive Precision Bridge):**
- Standalone: `researchgravity/cpb/`
- Integration: Python import, CLI
- Usage: ResearchGravity, meta-vengine

**VoiceNexus:**
- Standalone: `@metaventionsai/voice-nexus` npm package
- Integration: TypeScript SDK
- Usage: OS-App voice features

**Meta-Learning:**
- Standalone: HTTP API (localhost:3847)
- Integration: Python import, TypeScript SDK, CLI
- Usage: ResearchGravity, OS-App, meta-vengine

**Common Pattern:**
1. Build standalone core
2. Create integration layers (confirmed, not optional)
3. Add UI components (confirmed, not optional)
4. Maintain architectural independence

---

## Summary

**What It Is:**
A standalone predictive service that forecasts session outcomes based on multi-dimensional correlation.

**How It Works:**
Learns from 1,231 historical data points to predict quality, success probability, optimal timing, and potential errors.

**How to Use:**
- Python apps: Direct import or HTTP
- TypeScript apps: SDK (recommended) or HTTP
- CLI: Standalone tools
- Any app: HTTP API

**Status:**
- Service: ✅ Production-ready (standalone)
- SDK: ✅ Complete (optional but confirmed for use)
- UI: 🔜 Phase 7 (optional but confirmed for implementation)

**Architecture:**
Standalone-first microservice with confirmed integration layers and UI.

---

**Maintained by:** Antigravity Ecosystem
**Pattern:** Standalone → SDK → UI (all confirmed)
**Status:** Production-ready with planned enhancements 🚀
