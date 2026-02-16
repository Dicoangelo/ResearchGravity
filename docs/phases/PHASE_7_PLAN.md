# Phase 7: UI Components (Optional OS-App Enhancement)

## Overview

Create React components to visualize Meta-Learning predictions in the OS-App interface. These are **optional UI enhancements** - the Meta-Learning Engine works standalone regardless.

**Status:** 🔜 Planning
**Architecture:** Standalone service → SDK → UI Components
**Dependency:** Meta-Learning Engine (standalone), Agent Core SDK (Phase 6)

---

## Philosophy

Following the **Antigravity Innovation Pattern**:

```
Standalone Service (Phases 1-5)
      ↓
Integration Layer (Phase 6: SDK)
      ↓
UI Components (Phase 7: Optional)
```

**The service works without these components.** Phase 7 just makes predictions visible in the UI.

---

## Components to Build

### 1. PredictionBadge

**Purpose:** Show quality prediction and success probability

**Location:** `~/OS-App/components/predictions/PredictionBadge.tsx`

**Design:**
```tsx
interface PredictionBadgeProps {
  quality: number;        // 1-5
  successRate: number;    // 0-1
  confidence: number;     // 0-1
  compact?: boolean;
}

// Visual:
// 🟢 4.2★ | 78% success | 64% confident
```

**Use Cases:**
- Task planner header
- Session initialization dialog
- Dashboard task cards

---

### 2. ErrorWarningPanel

**Purpose:** Display potential errors with prevention strategies

**Location:** `~/OS-App/components/predictions/ErrorWarningPanel.tsx`

**Design:**
```tsx
interface ErrorWarningPanelProps {
  errors: ErrorPattern[];
  onDismiss?: (errorType: string) => void;
  compact?: boolean;
}

// Visual:
// ⚠️ Potential Errors (2)
//
// 🔴 GIT_USERNAME_MISMATCH - 95% preventable
// 💡 Solution: Configure git user.name before cloning
//
// 🟡 PERMISSION_DENIED - 90% preventable
// 💡 Solution: Check file permissions with ls -la
```

**Use Cases:**
- Pre-task warnings
- Agent Control Center alerts
- Command Palette suggestions

---

### 3. OptimalTimeIndicator

**Purpose:** Show best time for task and current status

**Location:** `~/OS-App/components/predictions/OptimalTimeIndicator.tsx`

**Design:**
```tsx
interface OptimalTimeIndicatorProps {
  optimalHour: number;
  currentHour: number;
  isOptimalNow: boolean;
  reasoning: string;
}

// Visual when optimal:
// ✅ Optimal Time (20:00)
// You're in the peak cognitive window

// Visual when NOT optimal:
// ⏳ Wait 5 hours (Optimal: 20:00)
// Current hour (15:00) has lower success rate for this task
```

**Use Cases:**
- Session scheduler
- Task queue manager
- Biometric panel integration

---

### 4. ResearchChips

**Purpose:** Display recommended research findings

**Location:** `~/OS-App/components/predictions/ResearchChips.tsx`

**Design:**
```tsx
interface ResearchChipsProps {
  research: SearchResult[];
  onSelect?: (result: SearchResult) => void;
  maxDisplay?: number;
}

// Visual:
// 📚 Recommended Research (3)
//
// [Multi-agent consensus (0.85)] [DQ scoring (0.73)] [+1 more]
// Click to inject into context
```

**Use Cases:**
- Knowledge Injector UI
- Research panel
- Context pack selector

---

### 5. PredictionPanel (Composite)

**Purpose:** Full prediction display combining all components

**Location:** `~/OS-App/components/predictions/PredictionPanel.tsx`

**Design:**
```tsx
interface PredictionPanelProps {
  intent: string;
  track?: boolean;
  showErrors?: boolean;
  showTiming?: boolean;
  showResearch?: boolean;
  onStartTask?: () => void;
}

// Visual:
// ┌─────────────────────────────────────┐
// │ 🔮 Session Prediction               │
// │                                     │
// │ 🟢 Quality: 4.2★ (78% success)     │
// │                                     │
// │ ⚠️ Potential Issues (2)             │
// │ • Git username mismatch             │
// │ • Permission denied                 │
// │                                     │
// │ ⏳ Optimal Time: 20:00 (5h wait)    │
// │                                     │
// │ 📚 Recommended Research (3)         │
// │ [Multi-agent] [DQ scoring] [+1]    │
// │                                     │
// │ [Start Now] [Schedule for Later]   │
// └─────────────────────────────────────┘
```

**Use Cases:**
- Dashboard main panel
- Task creation dialog
- Session planning view

---

### 6. SignalBreakdown (Advanced)

**Purpose:** Show correlation signal breakdown for power users

**Location:** `~/OS-App/components/predictions/SignalBreakdown.tsx`

**Design:**
```tsx
interface SignalBreakdownProps {
  signals: {
    outcome_score: number;
    cognitive_alignment: number;
    research_availability: number;
    error_probability: number;
  };
  showWeights?: boolean;
}

// Visual:
// 📊 Prediction Signals
//
// Outcome Match     ████████░░ 0.82 (50% weight)
// Cognitive Fit     ███████░░░ 0.75 (30% weight)
// Research Ready    ██████░░░░ 0.60 (15% weight)
// Error Risk        ████░░░░░░ 0.15 (5% penalty)
//
// Combined Score: 0.64 (64% confidence)
```

**Use Cases:**
- Advanced settings panel
- Calibration dashboard
- Developer tools

---

## Integration Points

### Dashboard.tsx

Add prediction panel to main dashboard:

```tsx
import { PredictionPanel } from './predictions/PredictionPanel';

function Dashboard() {
  const [taskIntent, setTaskIntent] = useState('');

  return (
    <div>
      <TaskInput onChange={setTaskIntent} />

      {taskIntent && (
        <PredictionPanel
          intent={taskIntent}
          track={true}
          showErrors={true}
          showTiming={true}
          showResearch={true}
        />
      )}

      {/* Existing dashboard content */}
    </div>
  );
}
```

### AgentControlCenter.tsx

Add error warnings to agent control:

```tsx
import { ErrorWarningPanel } from './predictions/ErrorWarningPanel';
import { useErrorPrediction } from '@antigravity/agent-core-sdk';

function AgentControlCenter() {
  const currentTask = useCurrentTask();
  const { errors } = useErrorPrediction({
    intent: currentTask?.description || '',
    preventableOnly: true
  });

  return (
    <div>
      {errors && errors.count > 0 && (
        <ErrorWarningPanel errors={errors.errors} />
      )}

      {/* Existing agent controls */}
    </div>
  );
}
```

### BiometricPanel.tsx

Integrate cognitive state with optimal timing:

```tsx
import { OptimalTimeIndicator } from './predictions/OptimalTimeIndicator';
import { useOptimalTime } from '@antigravity/agent-core-sdk';

function BiometricPanel() {
  const cognitiveState = useCognitiveState();
  const currentTask = useCurrentTask();

  const { optimalTime } = useOptimalTime({
    intent: currentTask?.description || '',
    currentHour: new Date().getHours()
  });

  return (
    <div>
      {/* Existing biometric displays */}

      {optimalTime && (
        <OptimalTimeIndicator
          optimalHour={optimalTime.optimal_hour}
          currentHour={new Date().getHours()}
          isOptimalNow={optimalTime.is_optimal_now}
          reasoning={optimalTime.reasoning}
        />
      )}
    </div>
  );
}
```

### knowledgeInjector.ts

Prioritize research based on predictions:

```typescript
import { agentCore } from '@antigravity/agent-core-sdk';

async function injectContextWithPrediction(query: string) {
  const { prediction, errors } = await agentCore.getPredictionWithContext(query, {
    includeErrors: true,
    includeOptimalTime: false
  });

  // Low success? Show warning and basic context
  if (prediction.success_probability < 0.7) {
    return {
      warning: `Low success probability (${(prediction.success_probability * 100).toFixed(0)}%)`,
      suggestedResearch: prediction.recommended_research,
      context: await getBasicContext(query)
    };
  }

  // High success? Prioritize recommended research
  const prioritizedContext = await selectContext(
    query,
    prediction.recommended_research.map(r => r.content)
  );

  return {
    context: prioritizedContext,
    prediction,
    errors: errors?.errors || []
  };
}
```

---

## Component Library Structure

```
~/OS-App/components/predictions/
├── index.ts                    # Export all components
├── PredictionBadge.tsx         # Quality + success indicator
├── ErrorWarningPanel.tsx       # Error prevention UI
├── OptimalTimeIndicator.tsx    # Timing suggestions
├── ResearchChips.tsx           # Recommended research
├── PredictionPanel.tsx         # Composite panel
├── SignalBreakdown.tsx         # Advanced signal display
└── styles/
    └── predictions.css         # Shared styles
```

**Export pattern:**
```typescript
// components/predictions/index.ts
export { PredictionBadge } from './PredictionBadge';
export { ErrorWarningPanel } from './ErrorWarningPanel';
export { OptimalTimeIndicator } from './OptimalTimeIndicator';
export { ResearchChips } from './ResearchChips';
export { PredictionPanel } from './PredictionPanel';
export { SignalBreakdown } from './SignalBreakdown';
```

---

## Styling Approach

Use existing OS-App design system:

```css
/* components/predictions/styles/predictions.css */

/* Quality badges */
.prediction-quality-high { color: #4caf50; }    /* Green for 4-5 */
.prediction-quality-medium { color: #ff9800; }  /* Orange for 3-4 */
.prediction-quality-low { color: #f44336; }     /* Red for 1-3 */

/* Success indicators */
.success-high { background: linear-gradient(135deg, #4caf50, #8bc34a); }
.success-medium { background: linear-gradient(135deg, #ff9800, #ffc107); }
.success-low { background: linear-gradient(135deg, #f44336, #e91e63); }

/* Error severity */
.error-high { border-left: 4px solid #f44336; }
.error-medium { border-left: 4px solid #ff9800; }

/* Timing indicators */
.timing-optimal { background: #e8f5e9; }
.timing-suboptimal { background: #fff3e0; }
```

---

## Implementation Plan

### Week 1: Core Components

**Day 1-2: PredictionBadge + ErrorWarningPanel**
- Build basic components
- Add to Dashboard.tsx
- Test with real API

**Day 3-4: OptimalTimeIndicator + ResearchChips**
- Timing UI
- Research recommendations
- Integrate with BiometricPanel

**Day 5: PredictionPanel (Composite)**
- Combine all components
- Add to task creation flow
- Polish UX

### Week 2: Integration & Polish

**Day 1-2: Knowledge Injector Enhancement**
- Priority research selection
- Warning displays
- Context optimization

**Day 3: AgentControlCenter Integration**
- Error warnings
- Prediction indicators
- Agent task optimization

**Day 4: SignalBreakdown (Advanced)**
- Power user view
- Calibration metrics
- Developer tools

**Day 5: Testing & Refinement**
- User acceptance testing
- Performance optimization
- Documentation

---

## Success Metrics

### Component Metrics

- [ ] All 6 components render without errors
- [ ] Components update in <100ms after prediction
- [ ] Proper loading states shown
- [ ] Error states handled gracefully

### Integration Metrics

- [ ] Dashboard shows predictions for task input
- [ ] Knowledge Injector prioritizes recommended research
- [ ] AgentControlCenter displays error warnings
- [ ] BiometricPanel shows optimal timing

### User Experience Metrics

- [ ] Predictions visible before task starts
- [ ] Warnings prevent >50% of predicted errors
- [ ] Users schedule tasks at optimal times
- [ ] Confidence in predictions >70%

---

## Optional Enhancements (Phase 8)

### Prediction History

Show past predictions vs actual outcomes:

```tsx
<PredictionHistory
  sessionId={sessionId}
  showAccuracy={true}
/>
```

### Calibration Dashboard

Admin view for prediction accuracy:

```tsx
<CalibrationDashboard
  days={30}
  showWeightRecommendations={true}
/>
```

### Prediction Notifications

Alert users when conditions improve:

```tsx
// Background service
if (prediction.is_optimal_now && !wasOptimalBefore) {
  notify("✅ Now is optimal time for: " + task);
}
```

---

## Testing Plan

### Unit Tests

```typescript
// __tests__/PredictionBadge.test.tsx
describe('PredictionBadge', () => {
  it('shows green for high quality (>4.0)', () => {
    render(<PredictionBadge quality={4.5} successRate={0.8} confidence={0.7} />);
    expect(screen.getByText(/4.5★/)).toHaveClass('prediction-quality-high');
  });

  it('shows orange for medium quality (3.0-4.0)', () => {
    render(<PredictionBadge quality={3.5} successRate={0.6} confidence={0.5} />);
    expect(screen.getByText(/3.5★/)).toHaveClass('prediction-quality-medium');
  });
});
```

### Integration Tests

```typescript
// __tests__/Dashboard.integration.test.tsx
describe('Dashboard with Predictions', () => {
  it('shows prediction panel when task entered', async () => {
    render(<Dashboard />);

    const input = screen.getByPlaceholderText('Enter task...');
    fireEvent.change(input, { target: { value: 'implement auth' } });

    await waitFor(() => {
      expect(screen.getByText(/Session Prediction/)).toBeInTheDocument();
    });
  });
});
```

### E2E Tests

```typescript
// e2e/predictions.spec.ts
test('prediction workflow', async ({ page }) => {
  await page.goto('http://localhost:3000');

  // Enter task
  await page.fill('[data-testid="task-input"]', 'implement authentication');

  // Wait for prediction
  await page.waitForSelector('[data-testid="prediction-panel"]');

  // Check quality displayed
  const quality = await page.textContent('[data-testid="quality-badge"]');
  expect(quality).toMatch(/\d\.\d★/);

  // Check errors shown
  const errors = await page.textContent('[data-testid="error-warnings"]');
  expect(errors).toBeTruthy();
});
```

---

## Documentation

### Component Storybook

```typescript
// components/predictions/PredictionBadge.stories.tsx
export default {
  title: 'Predictions/PredictionBadge',
  component: PredictionBadge
};

export const HighQuality = {
  args: {
    quality: 4.5,
    successRate: 0.85,
    confidence: 0.72
  }
};

export const MediumQuality = {
  args: {
    quality: 3.2,
    successRate: 0.60,
    confidence: 0.55
  }
};

export const LowQuality = {
  args: {
    quality: 2.1,
    successRate: 0.40,
    confidence: 0.38
  }
};
```

### README Update

Add to `~/OS-App/README.md`:

```markdown
## Prediction Features

The OS-App integrates with the Meta-Learning Engine to provide predictive guidance:

- **Quality Predictions**: See predicted session quality before starting
- **Error Prevention**: Get warnings about potential errors with solutions
- **Optimal Timing**: Know when to work on specific tasks
- **Research Recommendations**: Prioritized research for your task

### Using Predictions

Predictions appear automatically when you:
1. Enter a task in the dashboard
2. Create a new agent session
3. Plan research sessions

See `components/predictions/` for component documentation.
```

---

## Deliverables

### Code

- [ ] 6 React components (PredictionBadge, ErrorWarningPanel, etc.)
- [ ] Component exports in index.ts
- [ ] Shared styles in predictions.css
- [ ] Integration in Dashboard, AgentControlCenter, BiometricPanel

### Tests

- [ ] Unit tests for each component
- [ ] Integration tests for Dashboard
- [ ] E2E test for prediction workflow

### Documentation

- [ ] Storybook stories for all components
- [ ] README update with prediction features
- [ ] PHASE_7_COMPLETE.md summary

---

## Summary

Phase 7 creates **optional UI components** to visualize Meta-Learning predictions in OS-App. These components consume the standalone service via the SDK (Phase 6).

**Key Points:**
- Meta-Learning Engine works without these components
- Components are OS-App enhancements only
- Other apps can build their own UI or use CLI/API directly
- Follows Antigravity Innovation Pattern: Standalone → SDK → UI

**Next Phase:** Phase 8 (optional) - Advanced features like prediction history, calibration dashboard, notifications

---

**Status:** 🔜 Ready to implement
**Estimated Duration:** 2 weeks
**Dependencies:** Meta-Learning Engine (Phases 1-5), SDK (Phase 6)
