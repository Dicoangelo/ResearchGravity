"""
Intelligence API Routes
=======================

REST API endpoints for the ResearchGravity intelligence layer.

Endpoints:
  GET  /api/v2/intelligence/status     - System capabilities
  POST /api/v2/intelligence/predict    - Unified prediction
  GET  /api/v2/intelligence/patterns   - Session patterns
  POST /api/v2/intelligence/feedback   - Outcome feedback
  GET  /api/v2/intelligence/optimal-time - Optimal time for tasks
  POST /api/v2/intelligence/errors     - Likely errors for context
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from intelligence import (
    predict_session_quality,
    get_likely_errors,
    get_related_research,
    get_optimal_time,
    get_session_patterns,
    get_status,
)

router = APIRouter(prefix="/api/v2/intelligence", tags=["intelligence"])


class PredictRequest(BaseModel):
    """Request for session quality prediction."""
    intent: str = Field(..., description="Task/intent description")
    context: Optional[str] = Field(None, description="Additional context")


class PredictResponse(BaseModel):
    """Response for session quality prediction."""
    intent: str
    predicted_quality: float
    success_probability: float
    optimal_hour: int
    cognitive_mode: str
    energy_level: float
    likely_errors: List[Dict[str, Any]]
    related_research: List[Dict[str, Any]]
    confidence: float
    timestamp: str


class ErrorsRequest(BaseModel):
    """Request for likely errors."""
    context: str = Field(..., description="Context to search for errors")
    limit: int = Field(5, description="Maximum errors to return")


class FeedbackRequest(BaseModel):
    """Request to record outcome feedback."""
    prediction_id: Optional[str] = Field(None, description="Original prediction ID")
    intent: str = Field(..., description="Original intent")
    actual_quality: float = Field(..., ge=1, le=5, description="Actual quality (1-5)")
    actual_outcome: str = Field(..., description="Outcome: success/partial/failed")
    session_id: Optional[str] = Field(None, description="Session ID")
    notes: Optional[str] = Field(None, description="Additional notes")


class FeedbackResponse(BaseModel):
    """Response for feedback recording."""
    success: bool
    message: str
    timestamp: str


@router.get("/status")
async def intelligence_status():
    """Get intelligence system status and capabilities."""
    try:
        status = await get_status()
        return {
            "status": "ok",
            "data": status,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictResponse)
async def predict_session(request: PredictRequest):
    """Predict session quality based on intent."""
    try:
        # Get storage engine if available
        storage_engine = None
        try:
            from storage.engine import get_engine
            storage_engine = await get_engine()
        except Exception:
            pass

        prediction = await predict_session_quality(
            request.intent,
            storage_engine
        )

        if storage_engine:
            await storage_engine.close()

        return PredictResponse(**prediction.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def session_patterns():
    """Get session patterns analysis."""
    try:
        patterns = get_session_patterns()
        return {
            "status": "ok",
            "data": patterns,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimal-time")
async def optimal_time(task_type: str = "general"):
    """Get optimal time for a task type."""
    try:
        result = get_optimal_time(task_type)
        return {
            "status": "ok",
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/errors")
async def likely_errors(request: ErrorsRequest):
    """Get likely errors for a context."""
    try:
        storage_engine = None
        try:
            from storage.engine import get_engine
            storage_engine = await get_engine()
        except Exception:
            pass

        errors = await get_likely_errors(request.context, storage_engine)

        if storage_engine:
            await storage_engine.close()

        return {
            "status": "ok",
            "data": {
                "context": request.context,
                "errors": errors[:request.limit],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(request: FeedbackRequest):
    """Record outcome feedback for calibration."""
    try:
        # Get storage engine
        storage_engine = None
        try:
            from storage.engine import get_engine
            storage_engine = await get_engine()
        except Exception:
            pass

        if storage_engine and request.prediction_id:
            try:
                await storage_engine.update_prediction_outcome(
                    prediction_id=request.prediction_id,
                    actual_quality=request.actual_quality,
                    actual_outcome=request.actual_outcome,
                    session_id=request.session_id or ""
                )
            except Exception as e:
                print(f"Warning: Could not update prediction: {e}")

        if storage_engine:
            await storage_engine.close()

        return FeedbackResponse(
            success=True,
            message="Feedback recorded",
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research")
async def related_research(query: str, limit: int = 5):
    """Find related research for a query."""
    try:
        storage_engine = None
        try:
            from storage.engine import get_engine
            storage_engine = await get_engine()
        except Exception:
            pass

        research = await get_related_research(query, storage_engine)

        if storage_engine:
            await storage_engine.close()

        return {
            "status": "ok",
            "data": {
                "query": query,
                "research": research[:limit],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
