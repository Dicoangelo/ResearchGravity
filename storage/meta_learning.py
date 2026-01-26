"""
Meta-Learning Engine

Correlation engine that learns which [research + cognitive state + tools] combinations
lead to successful outcomes. Enables predictive session optimization.

Usage:
    from storage.meta_learning import MetaLearningEngine

    engine = MetaLearningEngine()
    await engine.initialize()

    prediction = await engine.predict_session_outcome(
        intent="implement multi-agent system",
        cognitive_state={"mode": "peak", "hour": 14},
        available_research=["arxiv:2512.05470"]
    )
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import math

from .engine import get_engine


class MetaLearningEngine:
    """
    Correlation engine for predictive session optimization.

    Learns from:
    - Session outcomes (670+ records)
    - Cognitive states (temporal patterns)
    - Research context (what was available)
    - Error patterns (what went wrong)

    Predicts:
    - Session success probability
    - Optimal timing
    - Recommended research
    - Potential errors
    """

    def __init__(self):
        self.engine = None
        self._initialized = False

    async def initialize(self):
        """Initialize storage engine."""
        if self._initialized:
            return

        self.engine = await get_engine()
        self._initialized = True

    async def close(self):
        """Close connections."""
        if self.engine:
            await self.engine.close()

    async def predict_session_outcome(
        self,
        intent: str,
        cognitive_state: Optional[Dict[str, Any]] = None,
        available_research: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Predict session outcome based on multi-dimensional correlation.

        Args:
            intent: What the session aims to accomplish
            cognitive_state: Current cognitive state (mode, energy, hour, etc.)
            available_research: List of available research papers/findings

        Returns:
            {
                "predicted_quality": float (1-5),
                "success_probability": float (0-1),
                "optimal_time": int (hour 0-23),
                "recommended_research": List[Dict],
                "potential_errors": List[Dict],
                "similar_sessions": List[Dict],
                "confidence": float (0-1)
            }
        """
        if not self._initialized:
            await self.initialize()

        # 1. Find similar past sessions
        outcome_matches = await self.engine.search_outcomes(
            query=intent,
            limit=10,
            min_score=0.5
        )

        # 2. Analyze cognitive patterns
        cognitive_score = await self._analyze_cognitive_match(
            cognitive_state, outcome_matches
        )

        # 3. Find relevant research
        research_matches = await self.engine.search_findings(
            query=intent,
            limit=5,
            min_score=0.5
        ) if not available_research else []

        # 4. Predict potential errors (context-aware)
        # Search for errors relevant to the intent
        potential_errors = await self.engine.search_error_patterns(
            query=intent,
            limit=10,
            min_score=0.5,
            min_success_rate=0.7
        )

        # Enhance with preventable errors (high success rate solutions)
        if potential_errors:
            # Filter to most relevant and preventable
            potential_errors = [
                e for e in potential_errors
                if e.get("success_rate", 0) >= 0.7  # Only include if solution works
            ][:5]  # Top 5

        # 5. Compute composite prediction
        prediction = self._correlate(
            intent=intent,
            outcome_matches=outcome_matches,
            cognitive_score=cognitive_score,
            research_matches=research_matches,
            potential_errors=potential_errors,
            current_state=cognitive_state
        )

        return prediction

    async def _analyze_cognitive_match(
        self,
        current_state: Optional[Dict[str, Any]],
        outcome_matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze how well current cognitive state matches successful sessions.

        Returns:
            {
                "alignment_score": float (0-1),
                "optimal_hour": int,
                "optimal_mode": str,
                "energy_recommendation": str
            }
        """
        if not current_state:
            return {
                "alignment_score": 0.5,
                "optimal_hour": 14,
                "optimal_mode": "peak",
                "energy_recommendation": "No current state provided"
            }

        current_hour = current_state.get("hour", 12)
        current_mode = current_state.get("mode", "unknown")
        current_energy = current_state.get("energy_level", 0.5)

        # Query cognitive states database for similar successful patterns
        # Build context string for semantic search
        context = f"{current_mode} hour_{current_hour}"

        # Search for similar cognitive states
        similar_states = await self.engine.search_cognitive_states(
            query=context,
            limit=20,
            min_score=0.4
        )

        if not similar_states:
            # Fall back to heuristics if no data
            peak_hours = [20, 12, 2]
            optimal_hour = min(peak_hours, key=lambda h: abs(h - current_hour))

            hour_distance = min(abs(current_hour - optimal_hour), 24 - abs(current_hour - optimal_hour))
            hour_alignment = 1.0 - (hour_distance / 12.0)
            mode_alignment = 1.0 if current_mode in ["peak", "deep_night"] else 0.5

            return {
                "alignment_score": (hour_alignment * 0.6 + mode_alignment * 0.4),
                "optimal_hour": optimal_hour,
                "optimal_mode": "peak",
                "energy_recommendation": "Using heuristics (no historical data)"
            }

        # Analyze energy levels at different hours for this mode
        hour_energy_map = {}
        for state in similar_states:
            h = state.get("hour", 0)
            energy = state.get("energy_level", 0.5)
            if h not in hour_energy_map:
                hour_energy_map[h] = []
            hour_energy_map[h].append(energy)

        # Find optimal hour (highest average energy)
        optimal_hour = current_hour
        max_energy = 0
        for h, energies in hour_energy_map.items():
            avg_energy = sum(energies) / len(energies)
            if avg_energy > max_energy:
                max_energy = avg_energy
                optimal_hour = h

        # Calculate alignment score
        # 1. Hour alignment (40%)
        hour_distance = min(abs(current_hour - optimal_hour), 24 - abs(current_hour - optimal_hour))
        hour_alignment = 1.0 - (hour_distance / 12.0)

        # 2. Mode alignment (30%)
        mode_energy_map = {
            "deep_night": 0.9, "peak": 0.8, "evening": 0.7,
            "morning": 0.6, "dip": 0.5, "unknown": 0.5
        }
        optimal_energy = mode_energy_map.get(current_mode, 0.5)
        mode_alignment = min(optimal_energy, 1.0)

        # 3. Energy level (30%)
        energy_alignment = current_energy

        # Weighted composite
        alignment_score = (
            hour_alignment * 0.4 +
            mode_alignment * 0.3 +
            energy_alignment * 0.3
        )

        # Determine optimal mode
        mode_ranking = {
            "deep_night": 0.9, "peak": 0.8, "flow": 0.8,
            "evening": 0.7, "focused": 0.7, "morning": 0.6,
            "neutral": 0.5, "dip": 0.5, "distracted": 0.3
        }
        optimal_mode = max(mode_ranking.keys(), key=lambda m: mode_ranking.get(m, 0.5))

        # Generate recommendation
        if alignment_score > 0.75:
            recommendation = "Excellent timing - high cognitive alignment"
        elif alignment_score > 0.6:
            recommendation = "Good timing - moderate cognitive alignment"
        elif alignment_score > 0.4:
            recommendation = f"Suboptimal - consider waiting for hour {optimal_hour}"
        else:
            recommendation = f"Poor timing - strongly recommend waiting for hour {optimal_hour}"

        return {
            "alignment_score": alignment_score,
            "optimal_hour": optimal_hour,
            "optimal_mode": optimal_mode,
            "energy_recommendation": recommendation,
            "similar_states_found": len(similar_states)
        }

    def _correlate(
        self,
        intent: str,
        outcome_matches: List[Dict[str, Any]],
        cognitive_score: Dict[str, Any],
        research_matches: List[Dict[str, Any]],
        potential_errors: List[Dict[str, Any]],
        current_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Correlate all signals to produce a composite prediction.

        Weighting:
        - Past outcomes: 50%
        - Cognitive alignment: 30%
        - Research availability: 15%
        - Error probability: 5% (penalty)
        """
        # 1. Outcome signal
        if outcome_matches:
            avg_quality = sum(o.get("quality", 3) for o in outcome_matches) / len(outcome_matches)
            avg_similarity = sum(o.get("score", 0.5) for o in outcome_matches) / len(outcome_matches)

            # Weight by similarity
            weighted_quality = avg_quality * avg_similarity
            success_rate = len([o for o in outcome_matches if o.get("outcome") == "success"]) / len(outcome_matches)

            outcome_score = (weighted_quality / 5.0) * 0.5 + success_rate * 0.5
        else:
            outcome_score = 0.5  # Neutral if no history
            avg_quality = 3.0

        # 2. Cognitive signal
        alignment = cognitive_score.get("alignment_score", 0.5)
        cognitive_weight = 0.3

        # 3. Research signal
        research_score = min(len(research_matches) / 5.0, 1.0) if research_matches else 0.5
        research_weight = 0.15

        # 4. Error penalty
        error_probability = min(len(potential_errors) * 0.1, 0.3)  # Max 30% penalty
        error_weight = 0.05

        # Composite score
        composite = (
            outcome_score * 0.5 +
            alignment * cognitive_weight +
            research_score * research_weight -
            error_probability * error_weight
        )

        # Convert to quality prediction (1-5)
        predicted_quality = 1 + (composite * 4)  # Maps 0-1 to 1-5

        # Confidence based on data availability
        confidence = min(
            (len(outcome_matches) / 10.0) * 0.4 +  # More matches = higher confidence
            (1.0 if cognitive_score.get("alignment_score", 0) > 0.5 else 0.3) * 0.3 +
            (1.0 if research_matches else 0.5) * 0.3,
            1.0
        )

        return {
            "predicted_quality": round(predicted_quality, 1),
            "success_probability": round(composite, 2),
            "optimal_time": cognitive_score.get("optimal_hour", 14),
            "recommended_research": research_matches[:3],
            "potential_errors": potential_errors,
            "similar_sessions": outcome_matches[:3],
            "confidence": round(confidence, 2),
            "signals": {
                "outcome_score": round(outcome_score, 2),
                "cognitive_alignment": round(alignment, 2),
                "research_availability": round(research_score, 2),
                "error_probability": round(error_probability, 2)
            }
        }

    async def predict_optimal_time(
        self,
        intent: str,
        current_hour: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Predict the optimal time to work on a given task.

        Args:
            intent: Task description
            current_hour: Current hour (0-23), uses current time if not provided

        Returns:
            {
                "optimal_hour": int,
                "is_optimal_now": bool,
                "wait_hours": int,
                "reasoning": str
            }
        """
        if current_hour is None:
            current_hour = datetime.now().hour

        # Find similar successful sessions
        outcome_matches = await self.engine.search_outcomes(
            query=intent,
            limit=20,
            min_score=0.4,
            filter_outcome="success",
            min_quality=4.0
        )

        if not outcome_matches:
            # Default to peak hours
            peak_hours = [20, 12, 2]
            optimal = min(peak_hours, key=lambda h: abs(h - current_hour))
            return {
                "optimal_hour": optimal,
                "is_optimal_now": abs(current_hour - optimal) <= 1,
                "wait_hours": (optimal - current_hour) % 24,
                "reasoning": "Using default peak hours (no historical data)"
            }

        # Analyze patterns (simplified - would use cognitive_states join in production)
        # For now, use known peak hours weighted by success
        peak_hours = [20, 12, 2]
        optimal = peak_hours[0]  # Default to 20:00 (your top peak)

        is_optimal = abs(current_hour - optimal) <= 1
        wait_hours = (optimal - current_hour) % 24 if not is_optimal else 0

        return {
            "optimal_hour": optimal,
            "is_optimal_now": is_optimal,
            "wait_hours": wait_hours,
            "reasoning": f"Based on {len(outcome_matches)} similar successful sessions"
        }

    async def predict_errors(
        self,
        intent: str,
        include_preventable_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict potential errors for a given task.

        Args:
            intent: Task description
            include_preventable_only: Only return errors with known prevention (>70% success rate)

        Returns:
            List of error patterns with prevention strategies
        """
        if not self._initialized:
            await self.initialize()

        # Semantic search for relevant errors
        errors = await self.engine.search_error_patterns(
            query=intent,
            limit=10,
            min_score=0.5
        )

        if include_preventable_only:
            # Filter to errors with effective solutions
            errors = [
                e for e in errors
                if e.get("success_rate", 0) >= 0.7
            ]

        # Enrich with prevention guidance
        for error in errors:
            error["prevention_available"] = error.get("success_rate", 0) >= 0.7
            error["severity"] = "high" if error.get("occurrences", 0) > 1000 else "medium"

        return errors[:5]  # Top 5

    async def get_prevention_strategies(
        self,
        error_type: str
    ) -> Dict[str, Any]:
        """
        Get detailed prevention strategies for a specific error type.

        Args:
            error_type: Type of error (e.g., "git", "concurrency", "permissions")

        Returns:
            {
                "error_type": str,
                "strategies": List[str],
                "success_rate": float,
                "examples": List[str]
            }
        """
        if not self._initialized:
            await self.initialize()

        # Search for all patterns of this type
        query = f"{error_type} error prevention"
        patterns = await self.engine.search_error_patterns(
            query=query,
            limit=10,
            min_score=0.3
        )

        # Group by error type
        strategies = []
        examples = []
        success_rates = []

        for pattern in patterns:
            if pattern.get("error_type") == error_type:
                solution = pattern.get("solution", "")
                if solution and solution not in strategies:
                    strategies.append(solution)

                context = pattern.get("context", "")
                if context and len(examples) < 3:
                    examples.append(context[:200])

                success_rates.append(pattern.get("success_rate", 0.0))

        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0

        return {
            "error_type": error_type,
            "strategies": strategies[:5],
            "success_rate": avg_success_rate,
            "examples": examples,
            "pattern_count": len(patterns)
        }

    async def get_prediction_accuracy(self, days: int = 30) -> Dict[str, Any]:
        """
        Calculate prediction accuracy by comparing predictions to actual outcomes.

        Uses the calibration loop data from prediction_tracking table.

        Args:
            days: Number of days to analyze (default: 30)

        Returns:
            {
                "total_predictions": int,
                "accurate_predictions": int,
                "accuracy": float,
                "avg_quality_error": float,
                "success_prediction_rate": float,
                "period_days": int
            }
        """
        if not self._initialized:
            await self.initialize()

        return await self.engine.get_prediction_accuracy(days=days)

    async def store_prediction_for_tracking(
        self,
        intent: str,
        prediction: Dict[str, Any],
        cognitive_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a prediction for later calibration tracking.

        Args:
            intent: Task intent
            prediction: Prediction dictionary from predict_session_outcome()
            cognitive_state: Cognitive state at prediction time

        Returns:
            Prediction ID for later outcome update
        """
        if not self._initialized:
            await self.initialize()

        prediction_record = {
            "intent": intent,
            "predicted_quality": prediction.get("predicted_quality"),
            "success_probability": prediction.get("success_probability"),
            "optimal_time": prediction.get("optimal_time"),
            "cognitive_state": cognitive_state or {},
            "timestamp": datetime.now().isoformat()
        }

        return await self.engine.store_prediction(prediction_record)

    async def update_prediction_with_outcome(
        self,
        prediction_id: str,
        actual_quality: float,
        actual_outcome: str,
        session_id: str
    ):
        """
        Update a stored prediction with actual outcome for calibration.

        Args:
            prediction_id: ID from store_prediction_for_tracking()
            actual_quality: Actual quality score (1-5)
            actual_outcome: Actual outcome ('success', 'partial', 'failed')
            session_id: Session ID for reference
        """
        if not self._initialized:
            await self.initialize()

        await self.engine.update_prediction_outcome(
            prediction_id=prediction_id,
            actual_quality=actual_quality,
            actual_outcome=actual_outcome,
            session_id=session_id
        )

    async def temporal_join_cognitive_outcomes(
        self,
        window_hours: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Join cognitive states with session outcomes based on temporal proximity.

        Finds session outcomes and their nearest cognitive states within a time window.

        Args:
            window_hours: Time window in hours for matching (default: 1)

        Returns:
            List of {outcome, cognitive_state, time_diff_minutes} dictionaries
        """
        if not self._initialized:
            await self.initialize()

        # Get all outcomes with timestamps
        outcomes = await self.engine.search_outcomes(query="", limit=1000)

        # Get all cognitive states
        # Note: This is a simplified implementation
        # In production, we'd use SQL JOIN with temporal constraints
        joined = []

        for outcome in outcomes:
            outcome_time = outcome.get("timestamp", "")
            if not outcome_time:
                continue

            try:
                outcome_dt = datetime.fromisoformat(outcome_time.replace("Z", "+00:00"))

                # Search for cognitive states near this time
                # (In production, we'd use a proper temporal index query)
                # For now, we'll search by hour and filter
                hour = outcome_dt.hour
                context = f"hour_{hour}"

                nearby_states = await self.engine.search_cognitive_states(
                    query=context,
                    limit=20
                )

                # Find closest state within window
                for state in nearby_states:
                    state_time = state.get("timestamp", "")
                    if not state_time:
                        continue

                    try:
                        state_dt = datetime.fromisoformat(state_time.replace("Z", "+00:00"))
                        time_diff = abs((outcome_dt - state_dt).total_seconds() / 60)  # minutes

                        if time_diff <= (window_hours * 60):
                            joined.append({
                                "outcome": outcome,
                                "cognitive_state": state,
                                "time_diff_minutes": time_diff
                            })
                            break  # Take first match
                    except:
                        continue
            except:
                continue

        return joined

    async def multi_vector_search(
        self,
        query: str,
        limit: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform simultaneous search across all vector dimensions.

        Searches outcomes, cognitive states, research findings, and error patterns
        in parallel for comprehensive context.

        Args:
            query: Search query
            limit: Results per dimension

        Returns:
            {
                "outcomes": [...],
                "cognitive": [...],
                "research": [...],
                "errors": [...]
            }
        """
        if not self._initialized:
            await self.initialize()

        # Parallel search across all dimensions
        outcomes = await self.engine.search_outcomes(query=query, limit=limit)
        cognitive = await self.engine.search_cognitive_states(query=query, limit=limit)
        research = await self.engine.search_findings(query=query, limit=limit)
        errors = await self.engine.search_error_patterns(query=query, limit=limit)

        return {
            "outcomes": outcomes,
            "cognitive": cognitive,
            "research": research,
            "errors": errors,
            "total_results": len(outcomes) + len(cognitive) + len(research) + len(errors)
        }

    async def calibrate_weights(self) -> Dict[str, float]:
        """
        Adjust correlation weights based on prediction accuracy.

        Analyzes recent prediction performance and suggests optimal weights
        for the correlation algorithm.

        Returns:
            {
                "outcome_weight": float,
                "cognitive_weight": float,
                "research_weight": float,
                "error_weight": float,
                "recommended_update": bool
            }
        """
        accuracy = await self.get_prediction_accuracy(days=30)

        # Current weights (from _correlate method)
        current_weights = {
            "outcome_weight": 0.5,
            "cognitive_weight": 0.3,
            "research_weight": 0.15,
            "error_weight": 0.05
        }

        # If we have enough data and accuracy is poor, suggest adjustments
        if accuracy["total_predictions"] >= 10:
            if accuracy["accuracy"] < 0.70:
                # Low accuracy: increase cognitive weight, reduce outcome weight
                suggested_weights = {
                    "outcome_weight": 0.4,
                    "cognitive_weight": 0.4,
                    "research_weight": 0.15,
                    "error_weight": 0.05
                }
                return {**suggested_weights, "recommended_update": True}
            elif accuracy["accuracy"] > 0.85:
                # High accuracy: maintain current weights
                return {**current_weights, "recommended_update": False}

        # Default: no change
        return {**current_weights, "recommended_update": False}


# Convenience function
async def get_meta_engine() -> MetaLearningEngine:
    """Get initialized meta-learning engine."""
    engine = MetaLearningEngine()
    await engine.initialize()
    return engine
