#!/usr/bin/env python3
"""
CPB Precision Mode v2 - Orchestrator

Research-grounded answer system with:
1. Tiered online search (ResearchGravity methodology)
2. Grounded generation (agents cite only retrieved sources)
3. MAR consensus (Multi-Agent Reflexion)
4. Targeted refinement (IMPROVE pattern)
5. Editorial frame (thesis/gap/innovation)

Execution Flow:
1. TIERED SEARCH: arXiv + Labs + News + GitHub + Internal
2. CONTEXT GROUNDING: Build citation-ready context
3. CASCADE EXECUTION: 7-agent with grounded prompts
4. MAR CONSENSUS: Persona-guided critique → synthesis
5. TARGETED REFINEMENT: Fix weakest DQ dimension
6. EDITORIAL FRAME: Enforce thesis/gap/innovation

Research Foundation:
- arXiv:2512.20845 (MAR) - Multi-Agent Reflexion
- arXiv:2502.18530 (IMPROVE) - Targeted refinement
- arXiv:2511.15755 (DQ) - Quality measurement
- arXiv:2508.17536 (Voting) - Consensus strategies
- ResearchGravity SKILL.md - Tiered search methodology
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from .types import (
    CPBPath, CPBPhase, CPBResult, CPBRequest, DQScore,
    PathSignals, ACE_AGENT_PERSONAS
)
from .precision_config import (
    PRECISION_CONFIG, PrecisionConfig,
    PRECISION_AGENT_PERSONAS, PRECISION_DQ_WEIGHTS,
    get_precision_agent_prompts, calculate_precision_dq
)
from .rg_adapter import RGAdapter, RGContext, rg_adapter
from .critic_verifier import CriticVerifier, VerificationResult, verify, format_critic_feedback
from .orchestrator import cpb as base_cpb
from .search_layer import (
    TieredSearchLayer, SearchContext, SearchResult,
    SourceTier, SourceCategory, search_tiered, get_search_layer
)

# LLM client import (lazy to avoid circular deps)
_llm_executor = None

def _get_executor():
    """Get or create agent executor."""
    global _llm_executor
    if _llm_executor is None:
        from .llm_client import AgentExecutor, LLMClient
        _llm_executor = AgentExecutor(LLMClient())
    return _llm_executor


# =============================================================================
# PRECISION RESULT
# =============================================================================

@dataclass
class PrecisionResult:
    """Result from precision mode execution."""
    # Core output
    output: str = ""
    confidence: float = 0.0

    # DQ metrics
    dq_score: float = 0.0
    validity: float = 0.0
    specificity: float = 0.0
    correctness: float = 0.0

    # Editorial frame (v2)
    thesis: str = ""
    gap: str = ""
    innovation_direction: str = ""

    # Verification
    verified: bool = False
    verification: Optional[VerificationResult] = None

    # Execution metadata
    path: CPBPath = CPBPath.CASCADE
    execution_time_ms: int = 0
    retry_count: int = 0
    agent_count: int = 7

    # Search results (v2)
    search_time_ms: int = 0
    tier1_count: int = 0
    tier2_count: int = 0
    tier3_count: int = 0
    total_sources_found: int = 0

    # Context
    context_used: str = ""
    context_sources: List[Dict[str, Any]] = field(default_factory=list)
    rg_connection_mode: str = "unknown"
    grounding_prompt: str = ""  # Citation-ready context (v2)

    # Sources cited
    sources: List[Dict[str, Any]] = field(default_factory=list)
    citations_found: int = 0
    citations_verified: int = 0

    # Warnings and feedback
    warnings: List[str] = field(default_factory=list)
    feedback_history: List[str] = field(default_factory=list)
    refinement_targets: List[str] = field(default_factory=list)  # Which DQ dims were targeted (v2)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'output': self.output,
            'confidence': self.confidence,
            'dq_score': self.dq_score,
            'validity': self.validity,
            'specificity': self.specificity,
            'correctness': self.correctness,
            'thesis': self.thesis,
            'gap': self.gap,
            'innovation_direction': self.innovation_direction,
            'verified': self.verified,
            'verification': self.verification.to_dict() if self.verification else None,
            'path': self.path.value,
            'execution_time_ms': self.execution_time_ms,
            'search_time_ms': self.search_time_ms,
            'retry_count': self.retry_count,
            'agent_count': self.agent_count,
            'tier1_count': self.tier1_count,
            'tier2_count': self.tier2_count,
            'tier3_count': self.tier3_count,
            'total_sources_found': self.total_sources_found,
            'rg_connection_mode': self.rg_connection_mode,
            'sources': self.sources,
            'citations_found': self.citations_found,
            'citations_verified': self.citations_verified,
            'warnings': self.warnings,
            'feedback_history': self.feedback_history,
            'refinement_targets': self.refinement_targets,
        }
        return result


@dataclass
class PrecisionStatus:
    """Real-time status for precision execution."""
    phase: str = "initializing"
    progress: int = 0
    current_step: str = ""
    retry_attempt: int = 0
    max_retries: int = 5
    current_dq: float = 0.0
    target_dq: float = 0.95
    elapsed_ms: int = 0
    message: str = ""


# =============================================================================
# PRECISION ORCHESTRATOR
# =============================================================================

class PrecisionOrchestrator:
    """
    Precision Mode v2 Orchestrator for 95%+ quality answers.

    Integrates:
    - Tiered online search (ResearchGravity methodology)
    - Grounded generation (agents cite only retrieved sources)
    - MAR consensus (Multi-Agent Reflexion)
    - Targeted refinement (IMPROVE pattern)
    - Editorial frame (thesis/gap/innovation)
    """

    def __init__(self, config: PrecisionConfig = PRECISION_CONFIG):
        self.config = config
        self.rg_adapter = rg_adapter
        self.search_layer = get_search_layer()
        self.critic_verifier = CriticVerifier()
        self._status_callback: Optional[Callable[[PrecisionStatus], None]] = None

    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        on_status: Optional[Callable[[PrecisionStatus], None]] = None
    ) -> PrecisionResult:
        """
        Execute precision mode v2 query.

        Flow:
        1. TIERED SEARCH: arXiv + Labs + News + GitHub + Internal
        2. CONTEXT GROUNDING: Build citation-ready context
        3. CASCADE EXECUTION: 7-agent with grounded prompts
        4. MAR CONSENSUS: Persona-guided critique → synthesis
        5. TARGETED REFINEMENT: Fix weakest DQ dimension
        6. EDITORIAL FRAME: Enforce thesis/gap/innovation

        Args:
            query: The query to answer
            context: Optional additional context
            on_status: Optional callback for status updates

        Returns:
            PrecisionResult with verified answer
        """
        self._status_callback = on_status
        start_time = time.time()

        result = PrecisionResult(
            path=CPBPath.CASCADE,
            agent_count=self.config.ace_config.agent_count
        )

        try:
            # =================================================================
            # PHASE 1: TIERED SEARCH (ResearchGravity methodology)
            # =================================================================
            self._update_status("searching", 5, "Tier 1: Searching arXiv, labs, industry...")
            search_context = await self._execute_tiered_search(query)

            result.search_time_ms = search_context.search_time_ms
            result.tier1_count = len(search_context.tier1_results)
            result.tier2_count = len(search_context.tier2_results)
            result.tier3_count = len(search_context.tier3_results)
            result.total_sources_found = len(search_context.results)

            self._update_status(
                "searching", 15,
                f"Found {result.total_sources_found} sources (T1:{result.tier1_count} T2:{result.tier2_count} T3:{result.tier3_count})"
            )

            # =================================================================
            # PHASE 2: CONTEXT GROUNDING
            # =================================================================
            self._update_status("grounding", 20, "Building citation-ready context...")
            grounding_prompt = search_context.get_grounding_prompt()
            result.grounding_prompt = grounding_prompt[:2000] + "..." if len(grounding_prompt) > 2000 else grounding_prompt

            # Also get RG context for internal learnings
            rg_context = await self._enrich_context(query, context)
            result.rg_connection_mode = rg_context.connection_mode.value
            result.warnings.extend(rg_context.warnings)

            # Merge all context
            enriched_context = self._build_grounded_context(
                user_context=context,
                search_context=search_context,
                rg_context=rg_context
            )
            result.context_used = enriched_context[:1000] + "..."

            # =================================================================
            # PHASE 3: CASCADE EXECUTION (with grounded prompts)
            # =================================================================
            self._update_status("executing", 30, "Running 7-agent cascade with grounded sources...")
            cascade_result = await self._execute_grounded_cascade(
                query, enriched_context, search_context
            )
            result.output = cascade_result.get('output', '')
            result.sources = cascade_result.get('sources', [])

            # =================================================================
            # PHASE 4: MAR CONSENSUS (Multi-Agent Reflexion)
            # =================================================================
            self._update_status("consensus", 50, "Running MAR consensus critique...")
            mar_result = await self._execute_mar_consensus(
                query, result.output, enriched_context, search_context
            )
            result.output = mar_result.get('output', result.output)

            # =================================================================
            # PHASE 5: VERIFICATION + TARGETED REFINEMENT (IMPROVE pattern)
            # =================================================================
            self._update_status("verifying", 60, "Running critic verification...")
            verification = await self._verify_result(
                result.output, result.sources, query, enriched_context
            )

            result.verification = verification
            result.dq_score = verification.dq_score
            result.validity = verification.validity
            result.specificity = verification.specificity
            result.correctness = verification.correctness
            result.citations_found = verification.citations_found
            result.citations_verified = verification.citations_verified

            # Targeted refinement loop (IMPROVE pattern)
            retry = 0
            while (
                verification.dq_score < self.config.dq_threshold and
                retry < self.config.max_retries
            ):
                retry += 1

                # Identify weakest dimension
                weakest_dim = self._identify_weakest_dimension(verification)
                result.refinement_targets.append(weakest_dim)

                self._update_status(
                    "refining", 60 + (retry * 8),
                    f"Refining {weakest_dim} (DQ: {verification.dq_score:.2f})",
                    retry_attempt=retry,
                    current_dq=verification.dq_score
                )

                # Generate targeted feedback
                feedback = self._generate_targeted_feedback(verification, weakest_dim)
                result.feedback_history.append(f"[{weakest_dim}] {feedback}")

                # Re-execute with targeted refinement
                cascade_result = await self._execute_grounded_cascade(
                    query, enriched_context, search_context, feedback=feedback
                )
                result.output = cascade_result.get('output', '')
                result.sources = cascade_result.get('sources', [])

                # Re-verify
                verification = await self._verify_result(
                    result.output, result.sources, query, enriched_context
                )

                result.verification = verification
                result.dq_score = verification.dq_score
                result.validity = verification.validity
                result.specificity = verification.specificity
                result.correctness = verification.correctness
                result.citations_found = verification.citations_found
                result.citations_verified = verification.citations_verified

            result.retry_count = retry
            result.verified = verification.passed
            result.confidence = verification.dq_score * 100

            # =================================================================
            # PHASE 6: EDITORIAL FRAME (thesis/gap/innovation)
            # =================================================================
            self._update_status("framing", 95, "Extracting editorial frame...")
            editorial = self._extract_editorial_frame(result.output)
            result.thesis = editorial.get('thesis', '')
            result.gap = editorial.get('gap', '')
            result.innovation_direction = editorial.get('innovation_direction', '')

            # Complete
            elapsed_ms = int((time.time() - start_time) * 1000)
            result.execution_time_ms = elapsed_ms

            self._update_status(
                "complete", 100,
                f"Done (DQ: {result.dq_score:.3f}, retries: {retry}, sources: {result.total_sources_found})",
                current_dq=result.dq_score
            )

        except Exception as e:
            import traceback
            result.warnings.append(f"Execution error: {str(e)}")
            result.warnings.append(traceback.format_exc())
            self._update_status("error", 0, f"Error: {str(e)}")

        return result

    # =========================================================================
    # PHASE 1: TIERED SEARCH
    # =========================================================================

    async def _execute_tiered_search(self, query: str) -> SearchContext:
        """
        Execute tiered search using ResearchGravity methodology.

        Tier 1: arXiv, labs, industry news (highest priority)
        Tier 2: GitHub, benchmarks, social (amplifiers)
        Tier 3: Internal learnings (context)
        """
        return await search_tiered(query, max_results=30)

    # =========================================================================
    # PHASE 2: CONTEXT GROUNDING
    # =========================================================================

    def _build_grounded_context(
        self,
        user_context: Optional[str],
        search_context: SearchContext,
        rg_context: RGContext,
        budget: int = 20000
    ) -> str:
        """
        Build grounded context with citation-ready sources.

        Agents can ONLY cite sources in this context.
        """
        parts = []
        remaining = budget

        # Grounding prompt (search results - highest priority)
        grounding = search_context.get_grounding_prompt()
        if grounding and remaining > 0:
            grounding_truncated = grounding[:min(len(grounding), remaining // 2)]
            parts.append(grounding_truncated)
            remaining -= len(grounding_truncated)

        # User context
        if user_context and remaining > 0:
            user_truncated = user_context[:min(len(user_context), remaining // 3)]
            parts.append(f"## Additional Context\n{user_truncated}")
            remaining -= len(user_truncated)

        # RG internal context
        if remaining > 0:
            rg_enriched = rg_context.to_enriched_context(remaining)
            if rg_enriched:
                parts.append(rg_enriched)

        return "\n\n".join(parts)

    # =========================================================================
    # PHASE 3: GROUNDED CASCADE
    # =========================================================================

    async def _execute_grounded_cascade(
        self,
        query: str,
        context: str,
        search_context: SearchContext,
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute cascade with grounded prompts.

        Agents are instructed to ONLY cite from search results.
        """
        # Build prompts with grounding instruction
        prompts = get_precision_agent_prompts(query, context)

        # Add grounding instruction to each prompt
        grounding_instruction = """
## CITATION RULES (CRITICAL)
1. You MUST cite sources using [N] format from the "Retrieved Sources" section above
2. You may ONLY cite sources that appear in the retrieved sources list
3. Every factual claim must have a citation
4. Do NOT make up sources or use sources not in the list
5. Include the source's signal (stars, date) when citing
"""

        for prompt in prompts:
            prompt['user_prompt'] = grounding_instruction + "\n\n" + prompt['user_prompt']
            prompt['full_prompt'] = grounding_instruction + "\n\n" + prompt['full_prompt']

        # Add feedback if retrying
        if feedback:
            for prompt in prompts:
                prompt['user_prompt'] += f"\n\n## REFINEMENT FEEDBACK\n{feedback}"
                prompt['full_prompt'] += f"\n\n## REFINEMENT FEEDBACK\n{feedback}"

        # Get executor
        executor = _get_executor()

        # Check if LLM client is available
        if not executor.llm_client.get_available_providers():
            return {
                'output': self._generate_stub_response(query, context, prompts, feedback),
                'sources': self._extract_sources_from_search(search_context),
                'prompts': prompts,
                'agent_count': len(prompts),
                'warning': 'No LLM providers available - using stub response'
            }

        try:
            # Execute all agents in parallel
            self._update_status("executing", 35, "Running 7 agents in parallel...")
            agent_responses = await executor.execute_agents(
                prompts,
                model='sonnet',
                max_tokens=2048
            )

            # Synthesize responses
            self._update_status("synthesizing", 45, "Synthesizing agent perspectives...")
            synthesis = await executor.synthesize_responses(
                query,
                agent_responses,
                model='sonnet'
            )

            # Extract sources from search context
            sources = self._extract_sources_from_search(search_context)

            return {
                'output': synthesis.content,
                'sources': sources,
                'prompts': prompts,
                'agent_count': len(prompts),
                'agent_responses': [
                    {
                        'agent': ar.agent_name,
                        'role': ar.role,
                        'confidence': ar.confidence,
                        'response_preview': ar.response[:200] + '...' if len(ar.response) > 200 else ar.response
                    }
                    for ar in agent_responses
                ],
            }

        except Exception as e:
            return {
                'output': self._generate_stub_response(query, context, prompts, feedback),
                'sources': self._extract_sources_from_search(search_context),
                'prompts': prompts,
                'agent_count': len(prompts),
                'error': str(e)
            }

    def _extract_sources_from_search(self, search_context: SearchContext) -> List[Dict[str, Any]]:
        """Extract sources from search results for citation tracking."""
        sources = []
        for result in search_context.get_top_results(15):
            sources.append({
                'type': result.category.value,
                'tier': result.tier.value,
                'url': result.url,
                'title': result.title,
                'signal': result.signal_string,
                'score': result.final_score,
            })
        return sources

    # =========================================================================
    # PHASE 4: MAR CONSENSUS
    # =========================================================================

    async def _execute_mar_consensus(
        self,
        query: str,
        initial_output: str,
        context: str,
        search_context: SearchContext
    ) -> Dict[str, Any]:
        """
        Execute MAR (Multi-Agent Reflexion) consensus pattern.

        From arXiv:2512.20845:
        1. Persona-guided critics analyze from different perspectives
        2. Each critic contributes critiques and corrections
        3. Debate coordinator synthesizes into final consensus
        """
        executor = _get_executor()

        if not executor.llm_client.get_available_providers():
            return {'output': initial_output}

        try:
            # Define critic personas (from MAR pattern)
            critic_prompts = [
                {
                    'agent': 'ValidityCritic',
                    'role': 'validity',
                    'system_prompt': "You are a validity critic. Analyze logical coherence and technical feasibility.",
                    'user_prompt': f"""Review this response for validity issues:

{initial_output}

Find:
1. Logical inconsistencies
2. Technically infeasible claims
3. Unsupported assertions

Output a brief critique with specific issues."""
                },
                {
                    'agent': 'EvidenceCritic',
                    'role': 'evidence',
                    'system_prompt': "You are an evidence critic. Verify all citations against sources.",
                    'user_prompt': f"""Review this response for evidence quality:

{initial_output}

Available sources:
{search_context.get_grounding_prompt()[:3000]}

Check:
1. Are citations accurate?
2. Do sources support claims?
3. Are there unsupported claims?

Output a brief critique with specific issues."""
                },
                {
                    'agent': 'ActionabilityCritic',
                    'role': 'actionability',
                    'system_prompt': "You are an actionability critic. Assess practical value and specificity.",
                    'user_prompt': f"""Review this response for actionability:

{initial_output}

Check:
1. Are recommendations specific?
2. Can a reader act on this?
3. Are there vague or generic statements?

Output a brief critique with specific issues."""
                },
            ]

            # Run critics in parallel
            critic_responses = await executor.execute_agents(
                critic_prompts,
                model='haiku',  # Use Haiku for critics (fast, cheap)
                max_tokens=512
            )

            # Synthesize critiques into improved output
            critiques = "\n\n".join([
                f"**{cr.agent_name}**: {cr.response}"
                for cr in critic_responses
            ])

            improvement_prompt = f"""Based on these critiques, improve the response:

ORIGINAL RESPONSE:
{initial_output}

CRITIQUES:
{critiques}

CONTEXT:
{context[:5000]}

Generate an improved response that addresses all valid critiques while maintaining accuracy.
Include thesis, gap analysis, and innovation direction."""

            improved = await executor.llm_client.complete(
                improvement_prompt,
                model='sonnet',
                max_tokens=3000
            )

            return {
                'output': improved.content,
                'critiques': critiques,
                'critic_count': len(critic_responses),
            }

        except Exception as e:
            return {'output': initial_output, 'error': str(e)}

    # =========================================================================
    # PHASE 5: TARGETED REFINEMENT (IMPROVE pattern)
    # =========================================================================

    def _identify_weakest_dimension(self, verification: VerificationResult) -> str:
        """
        Identify weakest DQ dimension for targeted refinement.

        From arXiv:2502.18530 (IMPROVE):
        Focus on one component at a time rather than global updates.
        """
        dimensions = {
            'validity': verification.validity,
            'specificity': verification.specificity,
            'correctness': verification.correctness,
        }
        return min(dimensions, key=dimensions.get)

    def _generate_targeted_feedback(
        self,
        verification: VerificationResult,
        target_dim: str
    ) -> str:
        """Generate feedback targeting the weakest dimension."""
        prompts = {
            'validity': """IMPROVE VALIDITY:
- Strengthen logical coherence
- Add supporting evidence for claims
- Remove technically infeasible statements
- Ensure reasoning chain is sound""",

            'specificity': """IMPROVE SPECIFICITY:
- Add concrete details and numbers
- Include specific examples from sources
- Replace vague statements with precise ones
- Add quantitative signals (stars, citations, dates)""",

            'correctness': """IMPROVE CORRECTNESS:
- Verify all claims against cited sources
- Remove unsupported statements
- Ensure citations match content
- Fix any factual errors""",
        }

        base_feedback = prompts.get(target_dim, "Improve overall quality.")

        # Add specific issues if available
        if verification.issues:
            issue_text = "\n".join([f"- {issue}" for issue in verification.issues[:3]])
            base_feedback += f"\n\nSpecific issues to address:\n{issue_text}"

        return base_feedback

    # =========================================================================
    # PHASE 6: EDITORIAL FRAME
    # =========================================================================

    def _extract_editorial_frame(self, output: str) -> Dict[str, str]:
        """
        Extract thesis/gap/innovation from output.

        ResearchGravity methodology requires:
        - Thesis: "X is happening because Y, which means Z"
        - Gap: What's missing from current landscape
        - Innovation Direction: Concrete next step
        """
        import re

        result = {
            'thesis': '',
            'gap': '',
            'innovation_direction': '',
        }

        # Try to extract thesis
        thesis_patterns = [
            r'##?\s*(?:Thesis|Summary|Key Finding)[:\s]*\n([^\n]+)',
            r'(?:In summary|The key finding is|We find that)[:\s]*([^.]+\.)',
            r'(?:^|\n)([A-Z][^.]+because[^.]+\.)',
        ]
        for pattern in thesis_patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
            if match:
                result['thesis'] = match.group(1).strip()
                break

        # Try to extract gap
        gap_patterns = [
            r'##?\s*(?:Gap|Missing|Limitation)[:\s]*\n([^\n]+)',
            r'(?:The gap is|What\'s missing|However)[:\s]*([^.]+\.)',
        ]
        for pattern in gap_patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
            if match:
                result['gap'] = match.group(1).strip()
                break

        # Try to extract innovation direction
        innovation_patterns = [
            r'##?\s*(?:Innovation|Recommendation|Next Step)[:\s]*\n([^\n]+)',
            r'(?:We recommend|The next step|Innovation opportunity)[:\s]*([^.]+\.)',
        ]
        for pattern in innovation_patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
            if match:
                result['innovation_direction'] = match.group(1).strip()
                break

        return result

    async def _enrich_context(
        self,
        query: str,
        context: Optional[str]
    ) -> RGContext:
        """
        Enrich context from ResearchGravity.

        Args:
            query: Query for context retrieval
            context: Existing context

        Returns:
            RGContext with learnings, packs, sessions
        """
        return await self.rg_adapter.get_context(
            query,
            limit=10,
            include_learnings=True,
            include_packs=True,
            include_sessions=True
        )

    def _merge_context(
        self,
        user_context: Optional[str],
        rg_context: RGContext,
        budget: int = 15000
    ) -> str:
        """
        Merge user context with RG context within budget.

        Args:
            user_context: User-provided context
            rg_context: Context from ResearchGravity
            budget: Maximum characters

        Returns:
            Merged context string
        """
        parts = []
        remaining = budget

        # User context first (highest priority)
        if user_context:
            user_truncated = user_context[:min(len(user_context), budget // 2)]
            parts.append(f"## User Context\n{user_truncated}")
            remaining -= len(user_truncated)

        # Add RG context
        rg_enriched = rg_context.to_enriched_context(remaining)
        if rg_enriched:
            parts.append(rg_enriched)

        return "\n\n".join(parts)

    async def _execute_cascade(
        self,
        query: str,
        context: str,
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute cascade with 7-agent ACE consensus.

        Runs all 7 agents in parallel, then synthesizes their responses.

        Args:
            query: The query
            context: Enriched context
            feedback: Optional feedback from previous attempt

        Returns:
            Dict with output and sources
        """
        # Build prompts
        prompts = get_precision_agent_prompts(query, context)

        # Add feedback if retrying
        if feedback:
            for prompt in prompts:
                prompt['user_prompt'] += f"\n\n## FEEDBACK FROM PREVIOUS ATTEMPT\n{feedback}"
                prompt['full_prompt'] += f"\n\n## FEEDBACK FROM PREVIOUS ATTEMPT\n{feedback}"

        # Get executor
        executor = _get_executor()

        # Check if LLM client is available
        if not executor.llm_client.get_available_providers():
            # Fall back to stub if no providers available
            return {
                'output': self._generate_stub_response(query, context, prompts, feedback),
                'sources': self._extract_sources(context),
                'prompts': prompts,
                'agent_count': len(prompts),
                'warning': 'No LLM providers available - using stub response'
            }

        try:
            # Execute all agents in parallel
            self._update_status("executing", 35, "Running 7 agents in parallel...")
            agent_responses = await executor.execute_agents(
                prompts,
                model='sonnet',  # Use Sonnet for agents (cost-effective)
                max_tokens=2048
            )

            # Synthesize responses
            self._update_status("synthesizing", 50, "Synthesizing agent perspectives...")
            synthesis = await executor.synthesize_responses(
                query,
                agent_responses,
                model='sonnet'  # Use Sonnet for synthesis too
            )

            # Extract sources from synthesis and context
            sources = self._extract_sources(context)
            sources.extend(self._extract_sources(synthesis.content))

            # Calculate total cost
            total_cost = sum(
                ar.llm_response.cost_usd
                for ar in agent_responses
                if ar.llm_response
            )
            total_cost += synthesis.cost_usd

            return {
                'output': synthesis.content,
                'sources': sources,
                'prompts': prompts,
                'agent_count': len(prompts),
                'agent_responses': [
                    {
                        'agent': ar.agent_name,
                        'role': ar.role,
                        'confidence': ar.confidence,
                        'response_preview': ar.response[:200] + '...' if len(ar.response) > 200 else ar.response
                    }
                    for ar in agent_responses
                ],
                'total_cost_usd': total_cost,
                'synthesis_model': synthesis.model,
            }

        except Exception as e:
            # Fall back to stub on error
            return {
                'output': self._generate_stub_response(query, context, prompts, feedback),
                'sources': self._extract_sources(context),
                'prompts': prompts,
                'agent_count': len(prompts),
                'error': str(e)
            }

    def _generate_stub_response(
        self,
        query: str,
        context: str,
        prompts: List[Dict[str, str]],
        feedback: Optional[str]
    ) -> str:
        """
        Generate stub response when LLM is unavailable.

        Used as fallback when no API keys are configured.
        """
        response_parts = [
            f"# Analysis: {query[:100]}",
            "",
            "## Summary",
            f"This query was analyzed by {len(prompts)} specialized agents:",
            ""
        ]

        for prompt in prompts:
            response_parts.append(f"- **{prompt['agent']}** ({prompt['role']}): Perspective considered")

        response_parts.extend([
            "",
            "## Key Points",
            "1. Evidence-based analysis conducted",
            "2. Multiple perspectives synthesized",
            "3. Citations required for all claims",
            "",
            "## Recommendations",
            "- Review agent perspectives for detailed analysis",
            "- Verify citations against sources",
            "",
            "*Note: LLM integration unavailable. Configure API keys to enable full functionality.*",
            "*Run: python3 -m cpb.llm_client --setup to configure.*"
        ])

        if feedback:
            response_parts.extend([
                "",
                "## Addressed Feedback",
                feedback
            ])

        return "\n".join(response_parts)

    def _extract_sources(self, context: str) -> List[Dict[str, Any]]:
        """Extract source references from context."""
        import re

        sources = []

        # arXiv patterns
        for match in re.finditer(r'arXiv:(\d{4}\.\d{4,5})', context, re.IGNORECASE):
            sources.append({
                'type': 'arxiv',
                'arxiv_id': match.group(1),
                'url': f'https://arxiv.org/abs/{match.group(1)}'
            })

        # Session patterns
        for match in re.finditer(r'([a-z]+-[a-z]+-\d{8}-\d{6})', context, re.IGNORECASE):
            sources.append({
                'type': 'session',
                'session_id': match.group(1)
            })

        return sources

    async def _verify_result(
        self,
        output: str,
        sources: List[Dict[str, Any]],
        query: str,
        context: str
    ) -> VerificationResult:
        """
        Run critic verification on result.

        Args:
            output: Response output
            sources: Sources used
            query: Original query
            context: Context used

        Returns:
            VerificationResult with DQ score
        """
        return await verify(output, sources, query, context)

    def _update_status(
        self,
        phase: str,
        progress: int,
        message: str,
        retry_attempt: int = 0,
        current_dq: float = 0.0
    ):
        """Update status and call callback if set."""
        if self._status_callback:
            status = PrecisionStatus(
                phase=phase,
                progress=progress,
                current_step=message,
                retry_attempt=retry_attempt,
                max_retries=self.config.max_retries,
                current_dq=current_dq,
                target_dq=self.config.dq_threshold,
                message=message
            )
            self._status_callback(status)

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            'mode': 'precision',
            'config': {
                'dq_threshold': self.config.dq_threshold,
                'max_retries': self.config.max_retries,
                'agent_count': self.config.ace_config.agent_count,
                'force_cascade': self.config.force_cascade,
                'critic_validation': self.config.critic_validation,
            },
            'rg_status': self.rg_adapter.get_status()
        }

    def get_agent_info(self) -> List[Dict[str, str]]:
        """Get information about precision agents."""
        return [
            {'name': a['name'], 'role': a['role']}
            for a in PRECISION_AGENT_PERSONAS
        ]


# =============================================================================
# SINGLETON & CONVENIENCE
# =============================================================================

precision_orchestrator = PrecisionOrchestrator()


async def execute_precision(
    query: str,
    context: Optional[str] = None,
    on_status: Optional[Callable[[PrecisionStatus], None]] = None
) -> PrecisionResult:
    """Execute precision mode query."""
    return await precision_orchestrator.execute(query, context, on_status)


def get_precision_status() -> Dict[str, Any]:
    """Get precision orchestrator status."""
    return precision_orchestrator.get_status()


# =============================================================================
# PRE/POST EXECUTION HOOKS
# =============================================================================

class PrecisionHooks:
    """
    Hooks for extending precision execution.

    Register callbacks to run before/after various stages.
    """

    def __init__(self):
        self._pre_context: List[Callable] = []
        self._post_context: List[Callable] = []
        self._pre_execute: List[Callable] = []
        self._post_execute: List[Callable] = []
        self._pre_verify: List[Callable] = []
        self._post_verify: List[Callable] = []

    def on_pre_context(self, callback: Callable):
        """Register callback before context enrichment."""
        self._pre_context.append(callback)

    def on_post_context(self, callback: Callable):
        """Register callback after context enrichment."""
        self._post_context.append(callback)

    def on_pre_execute(self, callback: Callable):
        """Register callback before cascade execution."""
        self._pre_execute.append(callback)

    def on_post_execute(self, callback: Callable):
        """Register callback after cascade execution."""
        self._post_execute.append(callback)

    def on_pre_verify(self, callback: Callable):
        """Register callback before verification."""
        self._pre_verify.append(callback)

    def on_post_verify(self, callback: Callable):
        """Register callback after verification."""
        self._post_verify.append(callback)

    async def run_hooks(self, hook_type: str, data: Any) -> Any:
        """Run all hooks of a given type."""
        hooks = getattr(self, f'_{hook_type}', [])
        for hook in hooks:
            if asyncio.iscoroutinefunction(hook):
                data = await hook(data)
            else:
                data = hook(data)
        return data


# Global hooks instance
precision_hooks = PrecisionHooks()
