"""
Visual Intelligence Layer configuration.

Reads from ~/.agent-core/config.json and environment variables.
Supports switchable model profiles across the full Google Gemini ecosystem.

Profiles:
  max       — gemini-3-pro-preview + gemini-3-pro-image-preview @ 4K, 5 iters
  balanced  — gemini-2.5-pro + gemini-3-pro-image-preview @ 2k, 3 iters
  fast      — gemini-3-flash-preview + gemini-2.5-flash-image @ 2k, 3 iters
  budget    — gemini-2.5-flash-lite + gemini-2.5-flash-image @ 1k, 2 iters
  custom    — user-defined models from config.json
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# ── Model Profiles ──────────────────────────────────────────────
# Each profile specifies VLM model, image gen model, resolution,
# iterations, and estimated cost per diagram.

PROFILES: Dict[str, dict] = {
    "max": {
        "vlm_model": "gemini-3-pro-preview",
        "image_model": "gemini-3-pro-image-preview",
        "image_resolution": "4k",
        "max_iterations": 5,
        "cost_budget_per_diagram": 0.50,
        "cost_budget_per_session": 5.00,
        "description": "Maximum quality — Gemini 3 Pro VLM + 4K image gen, 5 critic rounds",
        "est_cost_per_diagram": 0.35,
        "est_time_seconds": 90,
    },
    "balanced": {
        "vlm_model": "gemini-2.5-pro",
        "image_model": "gemini-3-pro-image-preview",
        "image_resolution": "2k",
        "max_iterations": 3,
        "cost_budget_per_diagram": 0.30,
        "cost_budget_per_session": 3.00,
        "description": "Best value — Gemini 2.5 Pro VLM + 3 Pro image gen, 3 rounds",
        "est_cost_per_diagram": 0.18,
        "est_time_seconds": 55,
    },
    "fast": {
        "vlm_model": "gemini-3-flash-preview",
        "image_model": "gemini-2.5-flash-image",
        "image_resolution": "2k",
        "max_iterations": 3,
        "cost_budget_per_diagram": 0.15,
        "cost_budget_per_session": 2.00,
        "description": "Fast — Gemini 3 Flash VLM + Flash image gen, 3 rounds",
        "est_cost_per_diagram": 0.06,
        "est_time_seconds": 30,
    },
    "budget": {
        "vlm_model": "gemini-2.5-flash-lite",
        "image_model": "gemini-2.5-flash-image",
        "image_resolution": "1k",
        "max_iterations": 2,
        "cost_budget_per_diagram": 0.08,
        "cost_budget_per_session": 1.00,
        "description": "Budget — Gemini 2.5 Flash Lite + Flash image, 2 rounds",
        "est_cost_per_diagram": 0.02,
        "est_time_seconds": 15,
    },
}

# All available models for custom profiles
AVAILABLE_VLM_MODELS = {
    "gemini-3-pro-preview": {
        "description": "Best reasoning & multimodal (Gemini 3 Pro)",
        "input_cost_per_1m": 2.00,
        "output_cost_per_1m": 12.00,
        "context_window": 1_000_000,
    },
    "gemini-2.5-pro": {
        "description": "Advanced reasoning (Gemini 2.5 Pro)",
        "input_cost_per_1m": 1.25,
        "output_cost_per_1m": 10.00,
        "context_window": 1_000_000,
    },
    "gemini-3-flash-preview": {
        "description": "Fast frontier-class (Gemini 3 Flash)",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
        "context_window": 1_000_000,
    },
    "gemini-2.5-flash": {
        "description": "Fast and efficient (Gemini 2.5 Flash)",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
        "context_window": 1_000_000,
    },
    "gemini-2.5-flash-lite": {
        "description": "Ultra-fast budget (Gemini 2.5 Flash Lite)",
        "input_cost_per_1m": 0.10,
        "output_cost_per_1m": 0.40,
        "context_window": 1_000_000,
    },
}

AVAILABLE_IMAGE_MODELS = {
    "gemini-3-pro-image-preview": {
        "description": "Best quality image generation (Gemini 3 Pro)",
        "resolutions": ["1k", "2k", "4k"],
        "cost_1k": 0.039,
        "cost_2k": 0.134,
        "cost_4k": 0.240,
    },
    "gemini-2.5-flash-image": {
        "description": "Fast image generation (Gemini 2.5 Flash)",
        "resolutions": ["1k", "2k"],
        "cost_1k": 0.019,
        "cost_2k": 0.067,
        "cost_4k": None,  # Not supported
    },
}


@dataclass
class VisualConfig:
    """Configuration for the visual intelligence layer."""
    enabled: bool = True
    profile: str = "balanced"  # max, balanced, fast, budget, custom
    google_api_key: Optional[str] = None
    vlm_model: str = "gemini-2.5-pro"
    image_model: str = "gemini-3-pro-image-preview"
    image_resolution: str = "2k"
    max_iterations: int = 3
    reference_store: str = "qdrant"
    min_reference_score: float = 0.6
    cost_budget_per_diagram: float = 0.30
    cost_budget_per_session: float = 3.00
    output_dir: str = "visual_assets"

    def apply_profile(self, profile_name: str):
        """Apply a named profile, overriding current settings."""
        if profile_name not in PROFILES:
            return
        p = PROFILES[profile_name]
        self.profile = profile_name
        self.vlm_model = p["vlm_model"]
        self.image_model = p["image_model"]
        self.image_resolution = p["image_resolution"]
        self.max_iterations = p["max_iterations"]
        self.cost_budget_per_diagram = p["cost_budget_per_diagram"]
        self.cost_budget_per_session = p["cost_budget_per_session"]

    def get_image_cost(self) -> float:
        """Get per-image cost for current resolution."""
        model_info = AVAILABLE_IMAGE_MODELS.get(self.image_model, {})
        cost_key = f"cost_{self.image_resolution}"
        return model_info.get(cost_key, 0.04)

    def get_vlm_info(self) -> dict:
        """Get VLM model info."""
        return AVAILABLE_VLM_MODELS.get(self.vlm_model, {})

    def estimate_diagram_cost(self) -> float:
        """Estimate cost for a single diagram generation."""
        vlm_info = self.get_vlm_info()
        input_cost = vlm_info.get("input_cost_per_1m", 1.25)
        output_cost = vlm_info.get("output_cost_per_1m", 10.00)
        # ~15K input tokens + ~2K output tokens per iteration
        vlm_cost = (15_000 * input_cost / 1_000_000 + 2_000 * output_cost / 1_000_000) * self.max_iterations
        image_cost = self.get_image_cost()
        return round(vlm_cost + image_cost, 4)

    def to_dict(self) -> dict:
        """Export config for logging/display."""
        return {
            "profile": self.profile,
            "vlm_model": self.vlm_model,
            "image_model": self.image_model,
            "image_resolution": self.image_resolution,
            "max_iterations": self.max_iterations,
            "cost_budget_per_diagram": self.cost_budget_per_diagram,
            "cost_budget_per_session": self.cost_budget_per_session,
            "estimated_cost_per_diagram": self.estimate_diagram_cost(),
        }


def get_visual_config() -> VisualConfig:
    """Load visual config from ~/.agent-core/config.json + env vars."""
    config = VisualConfig()

    # Load from config.json
    config_path = Path.home() / ".agent-core" / "config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            pb = data.get("paperbanana", {})
            config.enabled = pb.get("enabled", True)

            # Apply profile first (if set), then allow individual overrides
            profile = pb.get("profile")
            if profile and profile in PROFILES:
                config.apply_profile(profile)

            # Individual field overrides (take precedence over profile)
            if "vlm_model" in pb:
                config.vlm_model = pb["vlm_model"]
            if "image_model" in pb:
                config.image_model = pb["image_model"]
            if "image_resolution" in pb:
                config.image_resolution = pb["image_resolution"]
            if "max_iterations" in pb:
                config.max_iterations = pb["max_iterations"]
            if "reference_store" in pb:
                config.reference_store = pb["reference_store"]
            if "min_reference_score" in pb:
                config.min_reference_score = pb["min_reference_score"]
            if "cost_budget_per_diagram" in pb:
                config.cost_budget_per_diagram = pb["cost_budget_per_diagram"]
            if "cost_budget_per_session" in pb:
                config.cost_budget_per_session = pb["cost_budget_per_session"]

            # Google API key: try gemini section first, then env
            gemini_cfg = data.get("gemini", {})
            if gemini_cfg.get("api_key"):
                config.google_api_key = gemini_cfg["api_key"]
        except (json.JSONDecodeError, IOError):
            pass

    # Environment overrides
    env_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY")
    if env_key:
        config.google_api_key = env_key

    if os.environ.get("PAPERBANANA_PROFILE"):
        config.apply_profile(os.environ["PAPERBANANA_PROFILE"])

    if os.environ.get("PAPERBANANA_MAX_ITERATIONS"):
        config.max_iterations = int(os.environ["PAPERBANANA_MAX_ITERATIONS"])

    if os.environ.get("PAPERBANANA_VLM_MODEL"):
        config.vlm_model = os.environ["PAPERBANANA_VLM_MODEL"]

    if os.environ.get("PAPERBANANA_IMAGE_MODEL"):
        config.image_model = os.environ["PAPERBANANA_IMAGE_MODEL"]

    if os.environ.get("PAPERBANANA_RESOLUTION"):
        config.image_resolution = os.environ["PAPERBANANA_RESOLUTION"]

    return config


def list_profiles() -> Dict[str, dict]:
    """List all available profiles with descriptions and cost estimates."""
    result = {}
    for name, p in PROFILES.items():
        result[name] = {
            **p,
            "vlm_info": AVAILABLE_VLM_MODELS.get(p["vlm_model"], {}),
            "image_info": AVAILABLE_IMAGE_MODELS.get(p["image_model"], {}),
        }
    return result


def list_models() -> dict:
    """List all available VLM and image generation models."""
    return {
        "vlm_models": AVAILABLE_VLM_MODELS,
        "image_models": AVAILABLE_IMAGE_MODELS,
    }
