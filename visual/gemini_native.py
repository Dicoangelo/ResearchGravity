"""
Gemini Native Image Generator — direct google-genai SDK integration.

This is the "source" image generation engine, using Google's gemini-3-pro-image-preview
model directly (same API used for logo generation, Miami scenes, brand assets).

Capabilities:
- 1K / 2K / 4K resolution output
- Configurable aspect ratios (1:1, 5:4, 9:16, 16:9, 3:4, 4:3)
- Multi-image input for editing/composition (up to 14 images)
- Streaming response for large outputs
- Google Search grounding (optional)
- Quality tiers (max, high, fast) matching brand asset pipeline
- Cost tracking per image and per session
- UCW-compatible asset metadata

Usage:
    from visual.gemini_native import GeminiImageGenerator

    gen = GeminiImageGenerator()
    result = await gen.generate(prompt="A research methodology diagram...", resolution="4K")
    # result = {"asset_id": "...", "png_path": "...", "metadata": {...}}

    # With reference images (editing/composition)
    result = await gen.generate(
        prompt="Combine these diagrams into a unified architecture",
        input_images=["path1.png", "path2.png"],
        resolution="4K",
        aspect_ratio="16:9",
    )
"""

import mimetypes
import os
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AVAILABLE_IMAGE_MODELS, VisualConfig, get_visual_config

# Lazy imports
GENAI_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    pass


# ── Cost table (per image, by resolution) ────────────────────────
IMAGE_COST = {
    "gemini-3-pro-image-preview": {"1K": 0.039, "2K": 0.134, "4K": 0.240},
    "gemini-2.5-flash-image": {"1K": 0.019, "2K": 0.067, "4K": None},
}

# Quality tier presets
QUALITY_TIERS = {
    "max": {
        "resolution": "4K",
        "suffix": (
            "\n\nOUTPUT REQUIREMENTS:\n"
            "- Render at absolute highest quality — 4K, print-ready\n"
            "- Ultra-sharp details, no blur or artifacts\n"
            "- Smooth gradients with no banding\n"
            "- Professional publication quality\n"
            "- Every pixel matters"
        ),
    },
    "high": {
        "resolution": "2K",
        "suffix": (
            "\n\nOUTPUT REQUIREMENTS:\n"
            "- High quality, sharp details\n"
            "- Clean composition, no artifacts\n"
            "- Professional quality output"
        ),
    },
    "fast": {
        "resolution": "1K",
        "suffix": "",
    },
}


class GeminiImageGenerator:
    """
    Direct Google Gemini image generation — the native, source-level API.

    Wraps google.genai.Client with:
    - Configurable resolution, aspect ratio, quality
    - Multi-image input (up to 14 reference images)
    - Streaming for large outputs
    - Optional Google Search grounding
    - Cost tracking and UCW metadata
    """

    MAX_INPUT_IMAGES = 14

    def __init__(
        self,
        config: Optional[VisualConfig] = None,
        api_key: Optional[str] = None,
    ):
        self.config = config or get_visual_config()
        self._api_key = api_key or self.config.google_api_key
        self._session_cost = 0.0
        self._client = None

    def _get_client(self):
        """Lazy-init the genai client."""
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai not installed. Run: pip install google-genai"
            )
        if not self._api_key:
            raise ValueError(
                "No Google API key. Set GEMINI_API_KEY or configure in ~/.agent-core/config.json"
            )
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _get_image_cost(self, model: str, resolution: str) -> float:
        """Get per-image cost for model + resolution."""
        costs = IMAGE_COST.get(model, {})
        cost = costs.get(resolution)
        if cost is None:
            return 0.04  # fallback estimate
        return cost

    async def generate(
        self,
        prompt: str,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        quality: Optional[str] = None,
        model: Optional[str] = None,
        input_images: Optional[List[str]] = None,
        use_search_grounding: bool = False,
        output_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an image using Gemini's native image generation.

        Args:
            prompt: Image description / generation prompt
            resolution: "1K", "2K", or "4K" (default from config or quality tier)
            aspect_ratio: "1:1", "5:4", "9:16", "16:9", "3:4", "4:3" (optional)
            quality: "max", "high", or "fast" (overrides resolution if set)
            model: Image model override (default: gemini-3-pro-image-preview)
            input_images: List of image paths for editing/composition (up to 14)
            use_search_grounding: Enable Google Search grounding for factual content
            output_dir: Directory to save output (default: visual_assets/)
            output_filename: Custom output filename
            session_id: Optional session ID for UCW capture

        Returns:
            Dict with asset_id, png_path, metadata, or error key on failure.
        """
        # Budget check
        if self._session_cost >= self.config.cost_budget_per_session:
            return {
                "error": f"Session budget exceeded (${self._session_cost:.2f} / ${self.config.cost_budget_per_session:.2f})",
                "skipped": True,
            }

        # Resolve quality tier
        tier = QUALITY_TIERS.get(quality, {}) if quality else {}
        raw_resolution = resolution or tier.get("resolution") or self.config.image_resolution
        effective_resolution = raw_resolution.upper()  # API expects "1K", "2K", "4K"
        effective_model = model or self.config.image_model
        quality_suffix = tier.get("suffix", "")

        # Validate resolution for model (config stores lowercase, API needs uppercase)
        model_info = AVAILABLE_IMAGE_MODELS.get(effective_model, {})
        supported_lower = model_info.get("resolutions", ["1k", "2k", "4k"])
        if effective_resolution.lower() not in supported_lower:
            effective_resolution = supported_lower[-1].upper() if supported_lower else "1K"

        # Build full prompt
        full_prompt = prompt + quality_suffix

        try:
            client = self._get_client()
        except (ImportError, ValueError) as e:
            return {"error": str(e)}

        asset_id = f"native-{uuid.uuid4().hex[:12]}"
        started_at = datetime.now()

        # Resolve output path
        out_dir = Path(output_dir) if output_dir else Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if output_filename:
            out_path = out_dir / output_filename
        else:
            timestamp = started_at.strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"{asset_id}_{timestamp}.png"

        # Build content parts
        parts = []

        # Load input images if provided
        input_count = 0
        if input_images:
            if len(input_images) > self.MAX_INPUT_IMAGES:
                return {
                    "error": f"Too many input images ({len(input_images)}). Maximum is {self.MAX_INPUT_IMAGES}."
                }
            for img_path in input_images:
                img_file = Path(img_path)
                if img_file.exists():
                    mime = mimetypes.guess_type(str(img_file))[0]
                    if mime and mime.startswith("image/"):
                        parts.append(
                            types.Part.from_bytes(
                                data=img_file.read_bytes(), mime_type=mime
                            )
                        )
                        input_count += 1

        # Add text prompt
        parts.append(types.Part.from_text(text=full_prompt))
        contents = [types.Content(role="user", parts=parts)]

        # Build generation config
        image_config_kwargs = {"image_size": effective_resolution}
        if aspect_ratio:
            image_config_kwargs["aspect_ratio"] = aspect_ratio

        gen_config_kwargs = {
            "response_modalities": ["IMAGE", "TEXT"],
            "image_config": types.ImageConfig(**image_config_kwargs),
        }

        # Optional Google Search grounding
        if use_search_grounding:
            gen_config_kwargs["tools"] = [
                types.Tool(google_search=types.GoogleSearch())
            ]

        generate_content_config = types.GenerateContentConfig(**gen_config_kwargs)

        # Generate via streaming
        try:
            image_saved = False
            model_text = ""

            for chunk in client.models.generate_content_stream(
                model=effective_model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.parts is None:
                    continue
                for part in chunk.parts:
                    if part.inline_data and part.inline_data.data:
                        # Save the image
                        image_data = part.inline_data.data
                        if isinstance(image_data, str):
                            import base64
                            image_data = base64.b64decode(image_data)

                        # Use PIL if available for format normalization
                        try:
                            from PIL import Image as PILImage
                            img = PILImage.open(BytesIO(image_data))
                            if img.mode == "RGBA":
                                rgb = PILImage.new("RGB", img.size, (255, 255, 255))
                                rgb.paste(img, mask=img.split()[3])
                                rgb.save(str(out_path), "PNG")
                            elif img.mode == "RGB":
                                img.save(str(out_path), "PNG")
                            else:
                                img.convert("RGB").save(str(out_path), "PNG")
                        except ImportError:
                            # Fallback: write raw bytes
                            out_path.write_bytes(image_data)

                        image_saved = True
                    elif part.text:
                        model_text += part.text

            if not image_saved:
                return {
                    "error": "No image generated in response",
                    "model_text": model_text or None,
                }

        except Exception as e:
            return {"error": f"Generation failed: {e}"}

        elapsed = (datetime.now() - started_at).total_seconds()
        cost = self._get_image_cost(effective_model, effective_resolution)
        self._session_cost += cost

        file_size = out_path.stat().st_size if out_path.exists() else 0

        return {
            "asset_id": asset_id,
            "png_path": str(out_path.resolve()),
            "session_id": session_id,
            "engine": "gemini_native",
            "model_text": model_text or None,
            "metadata": {
                "model": effective_model,
                "resolution": effective_resolution,
                "aspect_ratio": aspect_ratio,
                "quality": quality or "default",
                "input_images": input_count,
                "use_search_grounding": use_search_grounding,
                "elapsed_seconds": round(elapsed, 1),
                "estimated_cost_usd": round(cost, 4),
                "file_size_bytes": file_size,
            },
            "created_at": started_at.isoformat(),
        }

    async def edit_image(
        self,
        prompt: str,
        input_images: List[str],
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        quality: Optional[str] = None,
        output_dir: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Edit/compose images — convenience wrapper around generate() with required input_images.

        Args:
            prompt: Edit instruction
            input_images: 1-14 image paths to edit/compose
            resolution: Output resolution
            aspect_ratio: Output aspect ratio
            quality: Quality tier
            output_dir: Output directory
            session_id: Optional session ID
        """
        if not input_images:
            return {"error": "edit_image requires at least one input image"}

        return await self.generate(
            prompt=prompt,
            input_images=input_images,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            quality=quality,
            output_dir=output_dir,
            session_id=session_id,
        )

    def get_session_cost(self) -> float:
        """Get total cost for current session."""
        return round(self._session_cost, 4)

    def reset_session_cost(self):
        """Reset session cost tracker."""
        self._session_cost = 0.0

    @staticmethod
    def available() -> bool:
        """Check if google-genai SDK is installed."""
        return GENAI_AVAILABLE

    @staticmethod
    def supported_resolutions(model: str = "gemini-3-pro-image-preview") -> List[str]:
        """Get supported resolutions for a model."""
        info = AVAILABLE_IMAGE_MODELS.get(model, {})
        return info.get("resolutions", ["1K", "2K", "4K"])

    @staticmethod
    def supported_aspect_ratios() -> List[str]:
        """All supported aspect ratios."""
        return ["1:1", "3:4", "4:3", "5:4", "9:16", "16:9"]
