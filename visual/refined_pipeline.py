"""
Refined Image Generation Pipeline — PaperBanana-style 5-agent architecture.

Implements the proven methodology from arXiv:2601.23265:

  Plan → Style → [Generate → Critique] × T rounds

Key insight: the Critic produces a REFINED TEXTUAL DESCRIPTION, not image edits.
Each iteration regenerates from scratch using the improved description.
This prevents the layout drift that plagues edit-based approaches.

Architecture:
  1. Planner Agent  — VLM converts source context → detailed textual description (P)
  2. Stylist Agent  — VLM enriches P with aesthetic guidelines → optimized P*
  3. Visualizer     — Image-Gen(P*) → I_t (fresh generation each round)
  4. Critic Agent   — VLM examines I_t vs source S → refined description P_{t+1}
  5. Repeat steps 3-4 for T rounds (default T=3)

Usage:
    from visual.refined_pipeline import RefinedPipeline

    pipeline = RefinedPipeline()
    result = await pipeline.generate(
        source_context="ResearchGravity is a 6-tier architecture...",
        caption="System architecture diagram showing all tiers",
        resolution="4K",
        aspect_ratio="16:9",
        iterations=3,
    )
"""

import json
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


# ── Aesthetic Style Guide (distilled from PaperBanana Appendix F) ────────
# This is baked in so every generation benefits from it without needing
# a reference corpus. Covers the 5 dimensions from the paper.

STYLE_GUIDE = """
# Visual Style Guide for Technical Architecture Diagrams

## 1. Color Palettes
- Use color to group logic, not just to decorate. Avoid fully saturated backgrounds.
- Background fills ("Zone Strategy"): Very light, desaturated pastels (opacity ~10-15%).
  - Cream/Beige (#F5F5DC) — warm, academic feel
  - Pale Blue/Ice (#E6F3FF) — clean, technical feel
  - Mint/Sage (#E0F2F1) — soft, organic feel
  - Pale Lavender (#F3E5F5) — distinctive, modern feel
  - Pale Rose (#FFF0F0) — attention, validation, critical paths
  - Pale Gold (#FFFDE7) — intelligence, processing, active computation
- Each major zone/tier MUST have a distinct pastel background color. Never reuse the same
  background for adjacent zones.
- Functional elements: Medium saturation preferred.
  - Common pairings: Blue/Orange, Green/Purple, Teal/Pink
  - Colors distinguish STATUS rather than component type
  - Active/trainable elements: warm tones (Red, Orange, Deep Pink)
  - Static/frozen elements: cool tones (Grey, Ice Blue, Cyan)
- High saturation reserved for: Error/Loss, Ground Truth, or final output highlights

## 2. Shapes & Containers — USE VISUAL METAPHORS
- "Softened Geometry": Sharp corners for data; rounded corners for processes
- Process Nodes: Rounded Rectangles (corner radius 5-10px), dominant shape (~60%)
- **Databases/Storage: 3D Cylinders** — always render databases as 3D cylinder icons.
  This is the universal visual metaphor. Never use flat rectangles for databases.
- **Network/Graph structures: Actual node-edge graph visualizations** — show interconnected
  dots/nodes with lines between them to represent knowledge graphs or network structures.
- **Validation/Security: Shield icons** — represent quality gates and critics with shield shapes.
- **Intelligence/Brain: Brain or neural network icon** — for AI processing, routing, orchestration.
- **Search/Retrieval: Magnifying glass icon** — for search, semantic lookup, retrieval.
- **Data Flow: Pipeline/funnel shapes** — for data ingestion and transformation stages.
- Grouping: Solid, light-colored container = global view; "zoomed-in" breakout for detail
- Borders: Solid for physical components; Dashed for logical stages, optional paths, scopes

## 3. Lines & Arrows — SHOW DATA FLOW BETWEEN ZONES
- **Every major zone MUST have labeled arrows showing what flows between them.**
  Example: "Raw signals" from Capture → Storage; "Embeddings" from Storage → Intelligence;
  "Validated output" from Critics → API. These inter-zone arrows are critical for readability.
- Orthogonal/Elbow (right angles): For internal module connections (implies precision)
- Curved/Bezier: For cross-zone data flow, feedback loops, high-level narrative
- Solid Black/Grey: Standard data flow (forward pass)
- Dashed Lines: "Auxiliary flow" — feedback loops, optional paths, async processes
- Arrow labels should be SHORT (2-4 words max) and describe what data moves, not how

## 4. Typography & Icons
- **Tier/Zone headers: LARGE and bold** (at least 2x the size of component labels).
  The viewer should be able to read the zone names at a glance without zooming.
- Labels (module names): Sans-Serif (Arial, Roboto, Helvetica). Bold for headers, Regular for details.
- **Statistics and key numbers: Highlighted with accent color or badge/pill shape.**
  Numbers like "11,579 nodes" or "27M tokens" should stand out visually.
- Variables (math): Serif (Times New Roman). If it's a variable, it must be Serif and Italicized.
- Use small icons alongside labels where appropriate (magnifying glass for search, shield for
  validation, cylinder for storage, brain for intelligence, terminal for CLI).

## 5. Layout & Composition
- **Avoid flat grids.** The best diagrams have visual hierarchy through SIZE VARIATION:
  the most important zone should be physically larger than peripheral zones.
- Use organic flow where possible — data flows naturally from left→right or top→bottom,
  with the critical path (the main data pipeline) being the most prominent visual line.
- Create depth through layering: background zones, mid-ground components, foreground highlights.
- Leave breathing room between zones — whitespace is not wasted space.
- Cross-cutting concerns (APIs, integrations) work well as thin strips on edges or bottom.
- Information density should be balanced — avoid overcrowding any single zone.

## 6. Common Pitfalls to Avoid
- **GRID MONOTONY**: All boxes same size in a flat grid = boring. Vary sizes by importance.
- "PowerPoint Default" look: Standard Blue/Orange presets with heavy black outlines
- Font Mixing: Using Times New Roman for "Encoder" labels (dated look)
- Inconsistent Dimensions: Mixing flat 2D and 3D isometric without reason
- Saturated backgrounds for grouping (distracts from content)
- Ambiguous arrows: Using the same line style for "Data Flow" and "Feedback Loop"
- **Missing inter-zone arrows**: If zones aren't connected by arrows, the diagram fails
  to communicate how data flows through the system.
- **Simulated code blocks**: NEVER render fake code, pseudo-code, or code editor views
  in sidebars. Use clean bulleted lists of function/tool names instead.
- Text hallucination: NEVER invent statistics, tool names, or labels not in the source
"""

# ── Agent Prompts (adapted from PaperBanana Appendix G) ──────────────────

PLANNER_PROMPT = """You are an expert Diagram Planning Agent who creates visually compelling,
publication-quality architecture diagrams. Your task is to convert a source description
into an extremely detailed textual specification that will be fed to an image generation model.

## INPUT
- **Source Context**: {source_context}
- **Figure Caption**: {caption}

## YOUR TASK
Create a rich, detailed textual description of a visually striking diagram.
Think like a professional information designer, not a PowerPoint user.

Your description MUST include:

### A. Visual Metaphors (NOT just boxes)
- **Databases** → describe as 3D cylinder icons (not flat rectangles)
- **Graphs/Networks** → describe as actual node-edge visualizations with connected dots
- **Validation/Security** → describe with shield icons
- **Intelligence/AI** → describe with brain or neural network icons
- **Search** → describe with magnifying glass icons
- **Data ingestion** → describe with funnel or pipeline shapes
- Use varied shapes — rounded rectangles for processes, cylinders for storage,
  diamonds for decisions, circles for agents/personas

### B. Inter-Zone Data Flow (CRITICAL)
- For EVERY pair of connected zones, describe a labeled arrow showing what data flows between them.
  Example: "A curved arrow labeled 'Raw signals' flows from Signal Capture zone into Storage zone"
- The data flow arrows are what make architecture diagrams readable. Without them,
  it's just a collection of disconnected boxes.

### C. Visual Hierarchy
- The most important/central zone should be described as PHYSICALLY LARGER than peripheral zones.
- Vary element sizes by importance — key stats should be in highlighted badges/pills.
- Zone headers should be described as large, bold text (much bigger than component labels).

### D. Spatial Layout
- Describe the overall flow direction (left→right, top→bottom, or radial).
- Specify which zones are adjacent to which, and how they're grouped.
- Cross-cutting concerns (APIs, integrations) should be thin strips along edges.

### E. Precise Labels (MANDATORY)
- Copy exact names, statistics, and terminology from the source context.
- NEVER invent, guess, or hallucinate labels, numbers, tool names, or component names.
- Every text label must appear verbatim in the Source Context.
- If the source says "11,579 nodes" — use exactly "11,579 nodes", not "~12K nodes".

### F. Color Zones
- Each major zone MUST have a distinct light pastel background.
- Specify exact colors: pale blue (#E6F3FF), pale green (#E8F5E9), pale rose (#FFF0F0),
  pale lavender (#F3E5F5), pale gold (#FFFDE7), pale mint (#E0F2F1).
- Never reuse the same background color for adjacent zones.

## CRITICAL RULES
- Be MAXIMALLY detailed. Vague specs produce bad images. Specify everything explicitly.
- Use ONLY information from the Source Context. Zero hallucination tolerance.
- Structure your description with numbered sections for each major zone.
- The overall background should be pure white.
- NEVER describe fake code blocks, pseudo-code, or code editor views. Use clean lists instead.

## OUTPUT
Provide ONLY the detailed textual description. No conversational text or explanations.
"""

STYLIST_PROMPT = """## ROLE
You are a Lead Visual Designer who makes technical diagrams look stunning and distinctive —
the kind that get shared on social media and cited in presentations.

## TASK
You are provided with a preliminary description of a diagram. Your task is to AGGRESSIVELY
upgrade its visual quality while preserving all factual content. Push for visual distinction.
A "safe grid of boxes" is a failure state — push for visual richness and hierarchy.

## STYLE GUIDELINES
{style_guide}

## PRELIMINARY DESCRIPTION
{description}

## SOURCE CONTEXT (for reference — do not alter factual content)
{source_context}

## YOUR INTERVENTIONS (apply ALL that improve the result)

### 1. Visual Metaphor Upgrade
- If the description uses flat rectangles for databases → change to 3D cylinders
- If the description mentions graphs/networks → add actual node-edge visualization
- If agents/personas are mentioned → add small character icons or emoji-style avatars
- If validation/critics are mentioned → add shield or checkpoint icons

### 2. Hierarchy & Size Variation
- If all elements are described as the same size → vary them. The central/most important
  zone should be 30-50% larger than peripheral zones.
- Zone headers should be explicitly described as LARGE bold text.
- Key statistics (node counts, session counts, etc.) should be in highlighted pill/badge shapes.

### 3. Flow & Connection Enrichment
- If inter-zone arrows are missing → ADD them with labels describing what data flows.
- If arrows are plain → make the main data pipeline arrows thicker/more prominent than secondary flows.
- Add subtle curved arrows for feedback loops (dashed lines for async/optional flows).

### 4. Color & Depth
- Ensure each zone has a DISTINCT pastel background. No two adjacent zones same color.
- Add subtle drop shadows or layering effects for depth where appropriate.
- Use color accents (not just backgrounds) to highlight key elements.

### 5. Anti-Monotony Check
- If the layout reads as a flat grid → introduce size variation, organic grouping, or radial layout.
- If all borders look the same → vary border styles (solid vs dashed vs dotted) by function.
- If it feels like a spreadsheet → add icons, 3D elements, or visual metaphors.

## RULES
1. **PRESERVE all factual content** — every label, statistic, and name stays exactly as specified.
2. **NEVER add information** not in the description or source context.
3. **NEVER remove components** — only enhance their visual presentation.
4. **NEVER simulate code blocks** — use clean bulleted lists for tool/function names.
5. Your changes are PURELY visual/aesthetic. Zero content modifications.

## OUTPUT
Output ONLY the final polished Detailed Description. No conversational text or explanations.
"""

CRITIC_PROMPT = """## ROLE
You are a Lead Visual Quality Critic for technical diagrams, evaluating both
CONTENT FIDELITY and VISUAL DESIGN QUALITY.

## TASK
Conduct a thorough critique of the generated diagram across two dimensions:
1. **Factual correctness** — does it match the source?
2. **Visual quality** — is it visually compelling, not just accurate?

You are provided with the Detailed Description that was used to generate this image.
If you identify areas for improvement, provide a revised version of the Detailed Description.

## CRITIQUE DIMENSIONS

### 1. Content Fidelity (HIGHEST PRIORITY)
- **Text QA**: Check for typographical errors, nonsensical text, unclear labels,
  gibberish, or garbled text. Suggest specific corrections. THIS IS THE #1 CHECK.
- **No Hallucination**: Verify that ALL text labels, statistics, tool names, and
  component names exist in the Source Context. Flag any invented content.
- **Duplicate Detection**: Check for repeated/duplicated elements. Flag and fix.
- **Completeness**: Are all major components from the source represented?
- **No Fake Code**: If the image contains simulated code blocks, pseudo-code, or
  code editor views, flag this and replace with clean bulleted lists in the revision.

### 2. Visual Design Quality
- **Inter-Zone Flow**: Are there labeled arrows between major zones showing data flow?
  If zones are disconnected boxes with no arrows, this is a MAJOR issue — add flow arrows.
- **Visual Monotony**: Are all elements the same size in a flat grid? If yes, suggest
  size variation — the most important zone should be larger.
- **Visual Metaphors**: Are databases shown as cylinders (not flat boxes)? Are graphs
  shown with actual nodes? Are there icons for agents/critics/search?
- **Hierarchy**: Can you read zone headers at a glance? Are key stats highlighted?
- **Spacing**: Are any areas cramped or overcrowded? Flag and suggest redistribution.
- **Sidebar Quality**: If there are sidebars (API lists, tool lists), are they clean
  bulleted lists? Or do they contain gibberish/truncated text? Fix any sidebar issues.

### 3. Overall Assessment
Rate the diagram 1-5 on each:
- Faithfulness (does it match the source?)
- Readability (can you understand the system at a glance?)
- Aesthetics (would you put this in a presentation or README?)

## INPUT DATA
- **Target Diagram**: [attached image]
- **Detailed Description**: {description}
- **Source Context**: {source_context}
- **Figure Caption**: {caption}

## REVISION APPROACH
Your revised description should primarily be MODIFICATIONS to the original, not a rewrite.
Focus changes on the specific problems identified. Keep everything that works well.
Be as detailed as possible in the revision — vague descriptions produce worse images.

## OUTPUT
Provide your response strictly in the following JSON format:
```json
{{
    "critic_suggestions": "Your detailed critique. If perfect, write 'No changes needed.'",
    "text_errors": ["specific text/spelling errors found"],
    "hallucinated_content": ["labels/stats not in source context"],
    "duplicated_elements": ["elements appearing more than once"],
    "layout_issues": ["spacing/layout/design problems"],
    "missing_flow_arrows": ["pairs of zones that should be connected but aren't"],
    "visual_quality_score": {{"faithfulness": 0, "readability": 0, "aesthetics": 0}},
    "revised_description": "The fully revised description. If no changes needed, write 'No changes needed.'"
}}
```
"""

# ── Per-image cost table ─────────────────────────────────────────────────
IMAGE_COST = {
    "gemini-3-pro-image-preview": {"1K": 0.039, "2K": 0.134, "4K": 0.240},
    "gemini-2.5-flash-image": {"1K": 0.019, "2K": 0.067, "4K": None},
}


class RefinedPipeline:
    """
    PaperBanana-style 5-agent image generation pipeline.

    Plan → Style → [Generate → Critique] × T rounds

    Uses Gemini VLM for planning/styling/critiquing and Gemini image gen
    for visualization. The critic produces a refined TEXTUAL DESCRIPTION
    and each round regenerates from scratch — no edit_image drift.
    """

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
            return 0.04
        return cost

    def _estimate_vlm_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate VLM call cost."""
        from .config import AVAILABLE_VLM_MODELS

        info = AVAILABLE_VLM_MODELS.get(self.config.vlm_model, {})
        input_cost = info.get("input_cost_per_1m", 2.00)
        output_cost = info.get("output_cost_per_1m", 12.00)
        return (input_tokens * input_cost / 1_000_000) + (
            output_tokens * output_cost / 1_000_000
        )

    async def _vlm_call(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        image_mime: str = "image/png",
    ) -> str:
        """Call the VLM (text or multimodal) and return text response."""
        client = self._get_client()

        parts = []
        if image_bytes:
            parts.append(
                types.Part.from_bytes(data=image_bytes, mime_type=image_mime)
            )
        parts.append(types.Part.from_text(text=prompt))

        contents = [types.Content(role="user", parts=parts)]

        config = types.GenerateContentConfig(
            response_modalities=["TEXT"],
            temperature=1.0,  # Paper uses temperature=1
        )

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=self.config.vlm_model,
            contents=contents,
            config=config,
        ):
            if chunk.parts:
                for part in chunk.parts:
                    if part.text:
                        response_text += part.text

        # Track VLM cost estimate
        est_input = len(prompt) // 4 + (len(image_bytes) // 100 if image_bytes else 0)
        est_output = len(response_text) // 4
        self._session_cost += self._estimate_vlm_cost(est_input, est_output)

        return response_text

    async def _generate_image(
        self,
        description: str,
        resolution: str,
        aspect_ratio: Optional[str],
        out_path: Path,
    ) -> bool:
        """Generate an image from a textual description. Returns True on success."""
        client = self._get_client()

        effective_model = self.config.image_model

        parts = [types.Part.from_text(text=description)]
        contents = [types.Content(role="user", parts=parts)]

        image_config_kwargs = {"image_size": resolution}
        if aspect_ratio:
            image_config_kwargs["aspect_ratio"] = aspect_ratio

        gen_config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(**image_config_kwargs),
        )

        image_saved = False
        for chunk in client.models.generate_content_stream(
            model=effective_model,
            contents=contents,
            config=gen_config,
        ):
            if chunk.parts is None:
                continue
            for part in chunk.parts:
                if part.inline_data and part.inline_data.data:
                    image_data = part.inline_data.data
                    if isinstance(image_data, str):
                        import base64

                        image_data = base64.b64decode(image_data)

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
                        out_path.write_bytes(image_data)

                    image_saved = True

        # Track image generation cost
        self._session_cost += self._get_image_cost(effective_model, resolution)
        return image_saved

    # ── Pipeline Stages ──────────────────────────────────────────────────

    async def plan(self, source_context: str, caption: str) -> str:
        """
        Planner Agent: Convert source context → detailed textual description.

        This is the most critical stage. A good plan prevents hallucination
        because it forces the system to explicitly enumerate every element
        before image generation.
        """
        prompt = PLANNER_PROMPT.format(
            source_context=source_context,
            caption=caption,
        )
        return await self._vlm_call(prompt)

    async def stylize(self, description: str, source_context: str) -> str:
        """
        Stylist Agent: Enrich description with aesthetic guidelines.

        Adds color palettes, shapes, typography, and layout refinements
        without altering the factual content.
        """
        prompt = STYLIST_PROMPT.format(
            style_guide=STYLE_GUIDE,
            description=description,
            source_context=source_context,
        )
        return await self._vlm_call(prompt)

    async def critique(
        self,
        image_path: Path,
        description: str,
        source_context: str,
        caption: str,
    ) -> Dict[str, Any]:
        """
        Critic Agent: Examine generated image against source context.

        Returns a dict with:
          - critic_suggestions: text critique
          - text_errors: list of spelling/text errors
          - hallucinated_content: list of invented labels
          - duplicated_elements: list of repeated elements
          - layout_issues: list of spacing/layout problems
          - revised_description: the improved textual description

        The key insight: the Critic produces a REVISED DESCRIPTION,
        not image modifications. The next round regenerates from scratch.
        """
        prompt = CRITIC_PROMPT.format(
            description=description,
            source_context=source_context,
            caption=caption,
        )

        # Read the generated image
        image_bytes = image_path.read_bytes()

        response = await self._vlm_call(prompt, image_bytes=image_bytes)

        # Parse JSON from response
        try:
            # Try to extract JSON from the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(response[json_start:json_end])
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: return the raw response as a suggestion
        return {
            "critic_suggestions": response,
            "text_errors": [],
            "hallucinated_content": [],
            "duplicated_elements": [],
            "layout_issues": [],
            "revised_description": description,  # Keep original if parsing fails
        }

    # ── Main Pipeline ────────────────────────────────────────────────────

    async def generate(
        self,
        source_context: str,
        caption: str,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        quality: Optional[str] = None,
        iterations: Optional[int] = None,
        output_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
        session_id: Optional[str] = None,
        input_images: Optional[List[str]] = None,
        skip_planning: bool = False,
        custom_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full PaperBanana-style pipeline: Plan → Style → [Generate → Critique] × T.

        Args:
            source_context: The source material to visualize (methodology, architecture, etc.)
            caption: What the diagram should show (communicative intent)
            resolution: "1K", "2K", or "4K" (default from config or quality tier)
            aspect_ratio: "1:1", "5:4", "9:16", "16:9", "3:4", "4:3"
            quality: "max", "high", or "fast" (overrides resolution/iterations)
            iterations: Number of Critique → Refine → Regenerate rounds (default: 3)
            output_dir: Directory for output files
            output_filename: Custom filename for final output
            session_id: Optional session ID for UCW capture
            input_images: Reference images for style (optional)
            skip_planning: Skip Planner+Stylist, use source_context directly as description
            custom_description: Use this description instead of running Planner

        Returns:
            Dict with asset_id, png_path, metadata, pipeline_trace, or error key.
        """
        # Budget check
        if self._session_cost >= self.config.cost_budget_per_session:
            return {
                "error": f"Session budget exceeded (${self._session_cost:.2f} / ${self.config.cost_budget_per_session:.2f})",
                "skipped": True,
            }

        # Resolve quality tier
        from .gemini_native import QUALITY_TIERS

        tier = QUALITY_TIERS.get(quality, {}) if quality else {}
        raw_resolution = (
            resolution or tier.get("resolution") or self.config.image_resolution
        )
        effective_resolution = raw_resolution.upper()
        T = iterations or self.config.max_iterations

        # Validate resolution
        model_info = AVAILABLE_IMAGE_MODELS.get(self.config.image_model, {})
        supported = model_info.get("resolutions", ["1k", "2k", "4k"])
        if effective_resolution.lower() not in supported:
            effective_resolution = supported[-1].upper() if supported else "1K"

        try:
            self._get_client()
        except (ImportError, ValueError) as e:
            return {"error": str(e)}

        asset_id = f"refined-{uuid.uuid4().hex[:12]}"
        started_at = datetime.now()
        pipeline_trace = []

        # Resolve output paths
        out_dir = Path(output_dir) if output_dir else Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = started_at.strftime("%Y%m%d_%H%M%S")

        # ── STAGE 1: Planner ────────────────────────────────────────────
        if custom_description:
            description = custom_description
            pipeline_trace.append({"stage": "planner", "status": "skipped (custom)"})
        elif skip_planning:
            description = source_context
            pipeline_trace.append({"stage": "planner", "status": "skipped"})
        else:
            plan_start = datetime.now()
            description = await self.plan(source_context, caption)
            plan_elapsed = (datetime.now() - plan_start).total_seconds()
            pipeline_trace.append({
                "stage": "planner",
                "elapsed_s": round(plan_elapsed, 1),
                "description_length": len(description),
            })

        # ── STAGE 2: Stylist ─────────────────────────────────────────────
        if not skip_planning:
            style_start = datetime.now()
            styled_description = await self.stylize(description, source_context)
            style_elapsed = (datetime.now() - style_start).total_seconds()
            pipeline_trace.append({
                "stage": "stylist",
                "elapsed_s": round(style_elapsed, 1),
                "description_length": len(styled_description),
            })
        else:
            styled_description = description
            pipeline_trace.append({"stage": "stylist", "status": "skipped"})

        # ── STAGES 3-4: Iterative [Generate → Critique] × T ─────────────
        current_description = styled_description
        iteration_results = []

        for t in range(T):
            iter_start = datetime.now()
            is_final = t == T - 1

            # Determine output path for this iteration
            if is_final and output_filename:
                iter_path = out_dir / output_filename
            elif is_final:
                iter_path = out_dir / f"{asset_id}_{timestamp}.png"
            else:
                iter_path = out_dir / f"{asset_id}_{timestamp}_iter{t + 1}.png"

            # ── Generate from description (fresh each round) ─────────
            gen_start = datetime.now()
            success = await self._generate_image(
                description=current_description,
                resolution=effective_resolution,
                aspect_ratio=aspect_ratio,
                out_path=iter_path,
            )
            gen_elapsed = (datetime.now() - gen_start).total_seconds()

            if not success:
                iteration_results.append({
                    "iteration": t + 1,
                    "status": "generation_failed",
                    "elapsed_s": round(gen_elapsed, 1),
                })
                # If generation fails, try with current description again
                continue

            iter_result = {
                "iteration": t + 1,
                "png_path": str(iter_path.resolve()),
                "gen_elapsed_s": round(gen_elapsed, 1),
                "file_size_bytes": iter_path.stat().st_size if iter_path.exists() else 0,
            }

            # ── Critique (skip on final iteration — just keep the image) ─
            if not is_final:
                critic_start = datetime.now()
                critique_result = await self.critique(
                    image_path=iter_path,
                    description=current_description,
                    source_context=source_context,
                    caption=caption,
                )
                critic_elapsed = (datetime.now() - critic_start).total_seconds()

                iter_result["critic_elapsed_s"] = round(critic_elapsed, 1)
                iter_result["critic_suggestions"] = critique_result.get(
                    "critic_suggestions", ""
                )
                iter_result["text_errors"] = critique_result.get("text_errors", [])
                iter_result["hallucinated_content"] = critique_result.get(
                    "hallucinated_content", []
                )
                iter_result["duplicated_elements"] = critique_result.get(
                    "duplicated_elements", []
                )
                iter_result["missing_flow_arrows"] = critique_result.get(
                    "missing_flow_arrows", []
                )
                iter_result["visual_quality_score"] = critique_result.get(
                    "visual_quality_score", {}
                )

                # Update description for next round
                revised = critique_result.get("revised_description", "")
                if revised and revised != "No changes needed." and len(revised) > 100:
                    current_description = revised
                    iter_result["description_refined"] = True
                else:
                    iter_result["description_refined"] = False
                    # Critic found no issues — can stop early
                    if critique_result.get("critic_suggestions", "").startswith(
                        "No changes"
                    ):
                        iter_result["early_stop"] = True
                        # Promote this iteration's image to final path
                        if output_filename:
                            final_path = out_dir / output_filename
                        else:
                            final_path = out_dir / f"{asset_id}_{timestamp}.png"
                        if iter_path != final_path:
                            import shutil
                            shutil.copy2(str(iter_path), str(final_path))
                            iter_path = final_path
                        iteration_results.append(iter_result)
                        break

            iter_result["total_elapsed_s"] = round(
                (datetime.now() - iter_start).total_seconds(), 1
            )
            iteration_results.append(iter_result)

            pipeline_trace.append({
                "stage": f"iteration_{t + 1}",
                "generated": success,
                "description_refined": iter_result.get("description_refined", False),
                "elapsed_s": iter_result["total_elapsed_s"],
            })

        # ── Build result ─────────────────────────────────────────────────
        total_elapsed = (datetime.now() - started_at).total_seconds()

        # Final image path is the last successful iteration
        final_result = None
        for r in reversed(iteration_results):
            if r.get("png_path"):
                final_result = r
                break

        if not final_result:
            return {
                "error": "All generation attempts failed",
                "pipeline_trace": pipeline_trace,
                "iteration_results": iteration_results,
            }

        final_path = final_result["png_path"]
        file_size = Path(final_path).stat().st_size if Path(final_path).exists() else 0

        return {
            "asset_id": asset_id,
            "png_path": final_path,
            "session_id": session_id,
            "engine": "refined_pipeline",
            "metadata": {
                "model": self.config.image_model,
                "vlm_model": self.config.vlm_model,
                "resolution": effective_resolution,
                "aspect_ratio": aspect_ratio,
                "quality": quality or "default",
                "iterations_run": len(iteration_results),
                "iterations_planned": T,
                "elapsed_seconds": round(total_elapsed, 1),
                "estimated_cost_usd": round(self._session_cost, 4),
                "file_size_bytes": file_size,
                "pipeline": "plan_style_critique",
            },
            "pipeline_trace": pipeline_trace,
            "iteration_results": iteration_results,
            "final_description": current_description,
            "created_at": started_at.isoformat(),
        }

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
