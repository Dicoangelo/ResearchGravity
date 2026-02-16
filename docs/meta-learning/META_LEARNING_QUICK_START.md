# Meta-Learning Engine: Quick Start Guide

## TL;DR

**Standalone HTTP service** that predicts session outcomes. Use from any app via API, Python, or TypeScript SDK.

```bash
# Start the service
python3 -m api.server --port 3847

# Use from ResearchGravity
python3 predict_session.py "implement feature X"

# Use from OS-App
import { AgentCoreClient } from '@antigravity/agent-core-sdk';
const prediction = await client.predictSession({ intent: 'feature X' });

# Use from meta-vengine
python3 -c "import httpx; ..."  # HTTP client
```

---

## Start the Service

```bash
cd ~/researchgravity
source .venv/bin/activate
python3 -m api.server --port 3847
```

**Service runs at:** `http://localhost:3847`
**Dependencies:** Qdrant (localhost:6333), SQLite, Cohere API key

**Verify:**
```bash
curl http://localhost:3847/
# Should return: {"service":"Antigravity Chief of Staff",...}
```

---

## Usage by Application

### From ResearchGravity (Python CLI)

**Use Case:** Predict session quality before initializing research

```bash
# Option 1: CLI tool
python3 predict_session.py "multi-agent orchestration" --hour 20 --verbose

# Output:
# 🔮 Session Outcome Prediction
# 🟢 Predicted Quality: 4.2/5 ⭐⭐⭐⭐
#    Success Probability: 78%
#    Optimal Time: 20:00
#    Similar Sessions: 3
#    Potential Errors: 2
```

```bash
# Option 2: HTTP client tool
python3 predict_api_client.py predict "implement auth" --track

# Option 3: Error prediction
python3 predict_errors.py "git clone repository"
```

**Integration:** Add to `init_session.py`:
```python
# Before creating session, check prediction
import subprocess
result = subprocess.run(
    ["python3", "predict_session.py", topic],
    capture_output=True
)
# Parse output and warn if low quality
```

---

### From OS-App (TypeScript + React)

**Use Case:** Show predictions in UI, enhance Knowledge Injector

**Option 1: TypeScript SDK (Recommended)**

```typescript
import { AgentCoreClient, useSessionPrediction } from '@antigravity/agent-core-sdk';

// Direct client usage
const client = new AgentCoreClient({ baseUrl: 'http://localhost:3847' });
const prediction = await client.predictSession({
  intent: 'implement authentication',
  cognitive_state: { mode: 'peak', hour: 20 },
  track_prediction: true
});

console.log(`Quality: ${prediction.predicted_quality}/5`);
console.log(`Success: ${prediction.success_probability * 100}%`);
```

```tsx
// React hook usage
function TaskPredictor({ taskIntent }: { taskIntent: string }) {
  const { prediction, isLoading, error } = useSessionPrediction({
    intent: taskIntent,
    cognitiveState: { mode: 'peak', hour: 20 },
    track: true,
    debounceMs: 500
  });

  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;
  if (!prediction) return null;

  return (
    <div>
      <QualityStars count={Math.round(prediction.predicted_quality)} />
      <SuccessRate value={prediction.success_probability} />

      {prediction.potential_errors.length > 0 && (
        <ErrorWarnings errors={prediction.potential_errors} />
      )}
    </div>
  );
}
```

**Option 2: Direct HTTP (No SDK required)**

```typescript
// Plain fetch
const response = await fetch('http://localhost:3847/api/v2/predict/session', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ intent: 'implement auth', track_prediction: true })
});
const prediction = await response.json();
```

**Integration Points:**
- `services/voiceNexus/knowledgeInjector.ts` - Prioritize research based on predictions
- `components/Dashboard.tsx` - Show prediction widgets
- `components/AgentControlCenter.tsx` - Display error warnings

---

### From meta-vengine (Python)

**Use Case:** Session optimization, cognitive routing

**Option 1: Direct Python Import**

```python
from storage.meta_learning import MetaLearningEngine

async def optimize_session(intent: str):
    engine = MetaLearningEngine()
    await engine.initialize()

    prediction = await engine.predict_session_outcome(
        intent=intent,
        cognitive_state=get_current_cognitive_state()
    )

    if prediction["success_probability"] < 0.6:
        print(f"⚠️  Low success probability: {prediction['success_probability']:.0%}")
        print(f"💡 Optimal time: {prediction['optimal_time']}:00")
        print(f"📚 Recommended research: {len(prediction['recommended_research'])} papers")
        return False

    return True

await engine.close()
```

**Option 2: HTTP Client**

```python
import httpx

async def get_prediction(intent: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:3847/api/v2/predict/session",
            json={"intent": intent, "track_prediction": True}
        )
        return response.json()

prediction = await get_prediction("implement feature")
```

**Integration:** Session optimizer script

```python
#!/usr/bin/env python3
import asyncio
import httpx

async def should_start_now(intent: str) -> bool:
    async with httpx.AsyncClient() as client:
        # Get prediction
        resp = await client.post(
            "http://localhost:3847/api/v2/predict/session",
            json={"intent": intent}
        )
        prediction = resp.json()

        # Check optimal time
        time_resp = await client.post(
            "http://localhost:3847/api/v2/predict/optimal-time",
            json={"intent": intent}
        )
        timing = time_resp.json()

        if not timing["is_optimal_now"]:
            print(f"⏳ Wait {timing['wait_hours']}h for optimal time")
            return False

        if prediction["success_probability"] < 0.7:
            print(f"⚠️  Success probability: {prediction['success_probability']:.0%}")
            return False

        return True

if __name__ == "__main__":
    intent = input("Task intent: ")
    ready = asyncio.run(should_start_now(intent))
    print("✅ Start now!" if ready else "🛑 Consider waiting")
```

---

### From Any Application (Generic HTTP)

**cURL:**

```bash
# Predict session
curl -X POST http://localhost:3847/api/v2/predict/session \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "implement authentication",
    "cognitive_state": {"mode": "peak", "hour": 20},
    "track_prediction": true
  }' | jq

# Predict errors
curl -X POST http://localhost:3847/api/v2/predict/errors \
  -H "Content-Type: application/json" \
  -d '{"intent": "git operations"}' | jq

# Get optimal time
curl -X POST http://localhost:3847/api/v2/predict/optimal-time \
  -H "Content-Type: application/json" \
  -d '{"intent": "architecture design", "current_hour": 15}' | jq

# Check accuracy
curl http://localhost:3847/api/v2/predict/accuracy?days=30 | jq
```

**Bash Script:**

```bash
#!/bin/bash
INTENT="$1"

# Get prediction
PREDICTION=$(curl -s -X POST http://localhost:3847/api/v2/predict/session \
  -H "Content-Type: application/json" \
  -d "{\"intent\": \"$INTENT\"}")

# Extract quality
QUALITY=$(echo "$PREDICTION" | jq -r '.predicted_quality')
SUCCESS=$(echo "$PREDICTION" | jq -r '.success_probability')

echo "Quality: $QUALITY/5"
echo "Success: $(echo "$SUCCESS * 100" | bc)%"

# Decision
if (( $(echo "$SUCCESS < 0.7" | bc -l) )); then
    echo "⚠️  Consider waiting for better conditions"
    exit 1
fi

echo "✅ Good to go!"
exit 0
```

---

## API Endpoints

All endpoints available at `http://localhost:3847/api/v2/predict/`:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/session` | Full session prediction |
| POST | `/errors` | Error prevention |
| POST | `/optimal-time` | Timing optimization |
| GET | `/accuracy?days=30` | Calibration metrics |
| POST | `/update-outcome` | Feedback loop |
| GET | `/multi-search?query=X&limit=5` | Multi-vector search |
| GET | `/calibrate-weights` | Weight recommendations |

**Request Examples:** See `META_LEARNING_ARCHITECTURE.md` for detailed API documentation.

---

## Integration Checklist

### For Python Applications (ResearchGravity, meta-vengine)

- [ ] Ensure Meta-Learning API is running (localhost:3847)
- [ ] Choose integration method:
  - [ ] Direct import: `from storage.meta_learning import MetaLearningEngine`
  - [ ] HTTP client: `httpx.AsyncClient()` or `requests`
  - [ ] CLI tools: Subprocess call
- [ ] Add prediction checks before critical operations
- [ ] Track outcomes for calibration

### For TypeScript Applications (OS-App)

- [ ] Ensure Meta-Learning API is running (localhost:3847)
- [ ] Choose integration method:
  - [ ] SDK (recommended): Install `@antigravity/agent-core-sdk`
  - [ ] Direct HTTP: Use `fetch()` API
- [ ] Add React hooks to components
- [ ] Display predictions in UI
- [ ] Track user interactions for calibration

---

## Common Use Cases

### 1. Pre-Session Quality Check

```bash
python3 predict_session.py "implement feature X" --verbose
# If quality < 3.0 or success < 60%, consider waiting
```

### 2. Error Prevention

```bash
python3 predict_errors.py "git clone private-repo"
# Shows: "git username mismatch - 95% preventable"
```

### 3. Optimal Timing

```bash
python3 predict_api_client.py optimal-time "architecture work" --hour 15
# Output: "Optimal hour: 20:00. Wait 5h for best results."
```

### 4. UI Predictions

```tsx
<TaskInput onChange={setIntent} />
<PredictionPanel intent={intent} />
// Shows quality/success badges in real-time
```

### 5. Session Optimization

```python
# Before starting intensive work
if await should_start_now(intent):
    start_session(intent)
else:
    schedule_for_later(intent)
```

---

## Troubleshooting

**Service won't start:**
```bash
# Check Qdrant is running
curl http://localhost:6333
# If not: docker run -p 6333:6333 qdrant/qdrant

# Check Cohere API key
cat ~/.agent-core/config.json | jq '.cohere.api_key'
# If missing: Add to config
```

**Predictions seem inaccurate:**
```bash
# Check calibration metrics
python3 predict_api_client.py accuracy --days 30

# If low accuracy, needs more tracked predictions
# Make predictions with track=true, then update outcomes
```

**SDK not found in OS-App:**
```bash
cd ~/OS-App/libs/agent-core-sdk
npm run build
npm link

cd ~/OS-App
npm link @antigravity/agent-core-sdk
```

---

## Summary

**Standalone Service:**
- Start with `python3 -m api.server --port 3847`
- Works independently of all consumers
- HTTP API accessible from any application

**Integration Methods:**
- **Python:** Direct import, HTTP client, or CLI tools
- **TypeScript:** SDK with React hooks, or direct fetch
- **Bash:** cURL commands or shell scripts

**Choose the integration method that fits your application best. The Meta-Learning Engine works the same way regardless.**

---

**Documentation:**
- Architecture: `META_LEARNING_ARCHITECTURE.md`
- Complete Summary: `META_LEARNING_ENGINE_COMPLETE.md`
- API Details: `PHASE_5_COMPLETE.md`
- SDK Details: `PHASE_6_COMPLETE.md`
