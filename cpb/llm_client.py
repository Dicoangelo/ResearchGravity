#!/usr/bin/env python3
"""
CPB Precision Mode - LLM Client

Unified LLM client supporting multiple providers:
- Anthropic Claude (preferred for precision mode)
- OpenAI GPT-4
- Google Gemini

Features:
- Automatic fallback between providers
- Parallel agent execution
- Token tracking and cost estimation
- Graceful error handling
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum

# Optional imports with fallbacks
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    openai = None
    HAS_OPENAI = False

try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GEMINI = True
except ImportError:
    genai = None
    genai_types = None
    HAS_GEMINI = False


# =============================================================================
# CONFIGURATION
# =============================================================================

HOME = Path.home()
CONFIG_FILE = HOME / ".agent-core" / "config.json"

# Model mappings
ANTHROPIC_MODELS = {
    'opus': 'claude-opus-4-20250514',
    'sonnet': 'claude-sonnet-4-20250514',
    'haiku': 'claude-3-5-haiku-20241022',
    'default': 'claude-sonnet-4-20250514',
}

OPENAI_MODELS = {
    'gpt4': 'gpt-4-turbo-preview',
    'gpt4o': 'gpt-4o',
    'gpt4o-mini': 'gpt-4o-mini',
    'default': 'gpt-4o',
}

GEMINI_MODELS = {
    'pro': 'gemini-1.5-pro',
    'flash': 'gemini-1.5-flash',
    'default': 'gemini-1.5-pro',
}

# Cost per 1M tokens (input/output)
MODEL_COSTS = {
    'claude-opus-4-20250514': (15.0, 75.0),
    'claude-sonnet-4-20250514': (3.0, 15.0),
    'claude-3-5-haiku-20241022': (0.25, 1.25),
    'gpt-4o': (2.5, 10.0),
    'gpt-4o-mini': (0.15, 0.6),
    'gemini-1.5-pro': (1.25, 5.0),
    'gemini-1.5-flash': (0.075, 0.3),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

class Provider(str, Enum):
    ANTHROPIC = 'anthropic'
    OPENAI = 'openai'
    GEMINI = 'gemini'


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    provider: Provider
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'model': self.model,
            'provider': self.provider.value,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'latency_ms': self.latency_ms,
            'cost_usd': self.cost_usd,
            'metadata': self.metadata,
        }


@dataclass
class AgentResponse:
    """Response from a single agent."""
    agent_name: str
    role: str
    response: str
    confidence: float = 0.0
    llm_response: Optional[LLMResponse] = None


# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

def load_api_keys() -> Dict[str, str]:
    """Load API keys from config file and environment."""
    keys = {}

    # Try config file first
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)

            # Direct keys
            for provider in ['anthropic', 'openai', 'gemini', 'cohere']:
                if provider in config:
                    if isinstance(config[provider], dict):
                        keys[provider] = config[provider].get('api_key', '')
                    else:
                        keys[provider] = config[provider]
        except Exception:
            pass

    # Override with environment variables
    env_mappings = {
        'anthropic': ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY'],
        'openai': ['OPENAI_API_KEY'],
        'gemini': ['GOOGLE_API_KEY', 'GEMINI_API_KEY'],
    }

    for provider, env_vars in env_mappings.items():
        for env_var in env_vars:
            if os.environ.get(env_var):
                keys[provider] = os.environ[env_var]
                break

    return keys


def save_api_key(provider: str, api_key: str):
    """Save an API key to config file."""
    config = {}

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
        except Exception:
            pass

    config[provider] = api_key

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    """
    Unified LLM client for precision mode.

    Supports Anthropic Claude, OpenAI GPT, and Google Gemini.
    Automatically falls back between providers on errors.
    """

    def __init__(
        self,
        preferred_provider: Provider = Provider.ANTHROPIC,
        api_keys: Optional[Dict[str, str]] = None
    ):
        self.preferred_provider = preferred_provider
        self.api_keys = api_keys or load_api_keys()
        self._clients: Dict[Provider, Any] = {}
        self._init_clients()

    def _init_clients(self):
        """Initialize available provider clients."""
        # Anthropic
        if HAS_ANTHROPIC and self.api_keys.get('anthropic'):
            try:
                self._clients[Provider.ANTHROPIC] = anthropic.Anthropic(
                    api_key=self.api_keys['anthropic']
                )
            except Exception as e:
                print(f"Warning: Could not initialize Anthropic client: {e}")

        # OpenAI
        if HAS_OPENAI and self.api_keys.get('openai'):
            try:
                self._clients[Provider.OPENAI] = openai.OpenAI(
                    api_key=self.api_keys['openai']
                )
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")

        # Gemini
        if HAS_GEMINI and self.api_keys.get('gemini'):
            try:
                genai.configure(api_key=self.api_keys['gemini'])
                self._clients[Provider.GEMINI] = genai
            except Exception as e:
                print(f"Warning: Could not initialize Gemini client: {e}")

    def get_available_providers(self) -> List[Provider]:
        """Get list of available providers."""
        return list(self._clients.keys())

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        provider: Optional[Provider] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generate a completion from an LLM.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User message
            model: Specific model to use (or None for default)
            provider: Specific provider (or None for preferred)
            max_tokens: Maximum response tokens
            temperature: Sampling temperature

        Returns:
            LLMResponse with content and metadata
        """
        # Determine provider order
        providers_to_try = []
        if provider and provider in self._clients:
            providers_to_try.append(provider)
        if self.preferred_provider in self._clients:
            providers_to_try.append(self.preferred_provider)
        providers_to_try.extend(self._clients.keys())

        # Deduplicate while preserving order
        seen = set()
        providers_to_try = [p for p in providers_to_try if not (p in seen or seen.add(p))]

        last_error = None
        for prov in providers_to_try:
            try:
                if prov == Provider.ANTHROPIC:
                    return await self._complete_anthropic(
                        system_prompt, user_prompt, model, max_tokens, temperature
                    )
                elif prov == Provider.OPENAI:
                    return await self._complete_openai(
                        system_prompt, user_prompt, model, max_tokens, temperature
                    )
                elif prov == Provider.GEMINI:
                    return await self._complete_gemini(
                        system_prompt, user_prompt, model, max_tokens, temperature
                    )
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    async def _complete_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """Complete using Anthropic Claude."""
        client = self._clients[Provider.ANTHROPIC]
        model_name = model or ANTHROPIC_MODELS['default']

        # Map short names to full names
        if model_name in ANTHROPIC_MODELS:
            model_name = ANTHROPIC_MODELS[model_name]

        start = time.time()

        # Run in thread pool since anthropic is sync
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
        )

        latency_ms = int((time.time() - start) * 1000)

        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Calculate cost
        costs = MODEL_COSTS.get(model_name, (3.0, 15.0))
        cost = (input_tokens * costs[0] + output_tokens * costs[1]) / 1_000_000

        return LLMResponse(
            content=content,
            model=model_name,
            provider=Provider.ANTHROPIC,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            metadata={'stop_reason': response.stop_reason}
        )

    async def _complete_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """Complete using OpenAI GPT."""
        client = self._clients[Provider.OPENAI]
        model_name = model or OPENAI_MODELS['default']

        # Map short names
        if model_name in OPENAI_MODELS:
            model_name = OPENAI_MODELS[model_name]

        start = time.time()

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
        )

        latency_ms = int((time.time() - start) * 1000)

        content = response.choices[0].message.content if response.choices else ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        costs = MODEL_COSTS.get(model_name, (2.5, 10.0))
        cost = (input_tokens * costs[0] + output_tokens * costs[1]) / 1_000_000

        return LLMResponse(
            content=content,
            model=model_name,
            provider=Provider.OPENAI,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            metadata={'finish_reason': response.choices[0].finish_reason if response.choices else None}
        )

    async def _complete_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """Complete using Google Gemini."""
        model_name = model or GEMINI_MODELS['default']

        if model_name in GEMINI_MODELS:
            model_name = GEMINI_MODELS[model_name]

        start = time.time()

        # Combine prompts for Gemini
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        loop = asyncio.get_event_loop()
        gemini_model = genai.GenerativeModel(model_name)

        response = await loop.run_in_executor(
            None,
            lambda: gemini_model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            )
        )

        latency_ms = int((time.time() - start) * 1000)

        content = response.text if response.text else ""

        # Estimate tokens (Gemini doesn't always return counts)
        input_tokens = len(full_prompt.split()) * 1.3  # Rough estimate
        output_tokens = len(content.split()) * 1.3

        costs = MODEL_COSTS.get(model_name, (1.25, 5.0))
        cost = (input_tokens * costs[0] + output_tokens * costs[1]) / 1_000_000

        return LLMResponse(
            content=content,
            model=model_name,
            provider=Provider.GEMINI,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    async def complete_parallel(
        self,
        prompts: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> List[LLMResponse]:
        """
        Execute multiple completions in parallel.

        Args:
            prompts: List of {'system_prompt': str, 'user_prompt': str}
            model: Model to use for all
            max_tokens: Max tokens per response
            temperature: Sampling temperature

        Returns:
            List of LLMResponses in same order as prompts
        """
        tasks = [
            self.complete(
                p['system_prompt'],
                p['user_prompt'],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            for p in prompts
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_status(self) -> Dict[str, Any]:
        """Get client status."""
        return {
            'preferred_provider': self.preferred_provider.value,
            'available_providers': [p.value for p in self.get_available_providers()],
            'has_anthropic': HAS_ANTHROPIC and 'anthropic' in self.api_keys,
            'has_openai': HAS_OPENAI and 'openai' in self.api_keys,
            'has_gemini': HAS_GEMINI and 'gemini' in self.api_keys,
        }


# =============================================================================
# AGENT EXECUTOR
# =============================================================================

class AgentExecutor:
    """
    Executes multi-agent consensus for precision mode.

    Runs agent prompts in parallel and synthesizes responses.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    async def execute_agents(
        self,
        agent_prompts: List[Dict[str, str]],
        model: str = 'sonnet',
        max_tokens: int = 2048,
    ) -> List[AgentResponse]:
        """
        Execute all agent prompts in parallel.

        Args:
            agent_prompts: List from get_precision_agent_prompts()
            model: Model to use (sonnet recommended for agents)
            max_tokens: Max tokens per agent response

        Returns:
            List of AgentResponses
        """
        # Build completion prompts
        prompts = [
            {
                'system_prompt': ap['system_prompt'],
                'user_prompt': ap['user_prompt']
            }
            for ap in agent_prompts
        ]

        # Execute in parallel
        # Use low temperature for precision mode consistency
        responses = await self.llm_client.complete_parallel(
            prompts,
            model=model,
            max_tokens=max_tokens,
            temperature=0.3  # Lower temp for consistent outputs
        )

        # Build agent responses
        agent_responses = []
        for i, (ap, resp) in enumerate(zip(agent_prompts, responses)):
            if isinstance(resp, Exception):
                # Handle failed agent
                agent_responses.append(AgentResponse(
                    agent_name=ap['agent'],
                    role=ap['role'],
                    response=f"[Agent failed: {str(resp)}]",
                    confidence=0.0,
                ))
            else:
                # Extract confidence from response if present
                confidence = self._extract_confidence(resp.content)
                agent_responses.append(AgentResponse(
                    agent_name=ap['agent'],
                    role=ap['role'],
                    response=resp.content,
                    confidence=confidence,
                    llm_response=resp
                ))

        return agent_responses

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence level from response text."""
        import re

        # Look for patterns like "Confidence: 85%" or "confidence level: 90"
        patterns = [
            r'[Cc]onfidence[:\s]+(\d+)%?',
            r'(\d+)%\s+confident',
            r'[Cc]onfidence\s+[Ll]evel[:\s]+(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                conf = int(match.group(1))
                return min(100, max(0, conf)) / 100

        return 0.7  # Default confidence

    async def synthesize_responses(
        self,
        query: str,
        agent_responses: List[AgentResponse],
        model: str = 'sonnet',
    ) -> LLMResponse:
        """
        Synthesize agent responses into final answer.

        Args:
            query: Original query
            agent_responses: Responses from all agents
            model: Model for synthesis

        Returns:
            Final synthesized response
        """
        # Build synthesis prompt
        agent_summaries = []
        for ar in agent_responses:
            conf_str = f"{ar.confidence*100:.0f}%" if ar.confidence else "N/A"
            agent_summaries.append(
                f"## {ar.agent_name} ({ar.role}) - Confidence: {conf_str}\n\n{ar.response}"
            )

        synthesis_prompt = f"""You are synthesizing insights from 7 specialized agents into a coherent, evidence-backed response.

## ORIGINAL QUERY
{query}

## AGENT PERSPECTIVES

{chr(10).join(agent_summaries)}

## SYNTHESIS INSTRUCTIONS

Create a comprehensive answer that:
1. **Synthesizes** key insights from all agents
2. **Cites sources** - use format [arXiv:XXXX.XXXXX] or [source_name]
3. **Addresses conflicts** - if agents disagreed, note the trade-offs
4. **Provides structure** - use headers, lists, and clear organization
5. **Includes recommendations** - actionable next steps

CRITICAL: Every factual claim must have a citation. If an agent made an unsupported claim, either find support or note it as "unverified".

Format your response with:
- ## Summary (2-3 sentences)
- ## Key Findings (bulleted, with citations)
- ## Analysis (detailed synthesis)
- ## Recommendations (actionable steps)
- ## Sources (list all citations)
"""

        return await self.llm_client.complete(
            system_prompt="You are a research synthesizer creating evidence-backed answers from multi-agent analysis.",
            user_prompt=synthesis_prompt,
            model=model,
            max_tokens=4096,
            temperature=0.2  # Very low temp for consistent synthesis
        )


# =============================================================================
# SINGLETON & CONVENIENCE
# =============================================================================

# Global client (lazy initialization)
_llm_client: Optional[LLMClient] = None
_agent_executor: Optional[AgentExecutor] = None


def get_llm_client() -> LLMClient:
    """Get or create LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def get_agent_executor() -> AgentExecutor:
    """Get or create agent executor singleton."""
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = AgentExecutor(get_llm_client())
    return _agent_executor


async def complete(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    **kwargs
) -> LLMResponse:
    """Convenience function for single completion."""
    return await get_llm_client().complete(system_prompt, user_prompt, model, **kwargs)


async def execute_agents(
    agent_prompts: List[Dict[str, str]],
    **kwargs
) -> List[AgentResponse]:
    """Convenience function for agent execution."""
    return await get_agent_executor().execute_agents(agent_prompts, **kwargs)


async def synthesize(
    query: str,
    agent_responses: List[AgentResponse],
    **kwargs
) -> LLMResponse:
    """Convenience function for synthesis."""
    return await get_agent_executor().synthesize_responses(query, agent_responses, **kwargs)
