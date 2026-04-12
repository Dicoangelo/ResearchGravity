"""
NotebookLM MCP Tools — HTTP/RPC + Cognitive Intelligence (37 tools)

NotebookLM Core (29):
  notebook_list, notebook_create, notebook_get, notebook_describe,
  notebook_rename, notebook_delete, chat_configure,
  source_add, source_describe, source_get_content,
  source_list_drive, source_sync_drive, source_delete,
  notebook_query, studio_create, studio_status, studio_delete,
  download_artifact, export_artifact,
  research_start, research_status, research_import,
  notebook_share_status, notebook_share_public, notebook_share_invite,
  note_create, note_list, note_update, note_delete

Cognitive Intelligence (6):
  cognitive_enrich_query, cognitive_search, cognitive_insights,
  research_to_notebook, knowledge_evolution, cognitive_auto_curate

Auth (2):
  save_auth_tokens, refresh_auth
"""

import json
from typing import Any, Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp_raw.logger import get_logger
from mcp_raw.protocol import tool_result_content, text_content

from ..api.client import NotebookLMAPIClient, AuthenticationError
from ..api.cognitive import CognitiveLayer
from ..api import constants

log = get_logger("notebooklm_tools")

# Module-level state (injected by server)
_api_client: Optional[NotebookLMAPIClient] = None
_cognitive: Optional[CognitiveLayer] = None
_pg_pool = None  # Set by server when DB is ready
_embedding_pipeline = None  # Set by server when embeddings are ready


def set_api_client(client: NotebookLMAPIClient):
    global _api_client
    _api_client = client


def set_cognitive(layer: CognitiveLayer):
    global _cognitive
    _cognitive = layer


def set_db_pool(pool, embedding_pipeline=None):
    global _pg_pool, _embedding_pipeline
    _pg_pool = pool
    _embedding_pipeline = embedding_pipeline
    # Auto-init cognitive layer in search-only mode when DB arrives
    _init_cognitive_if_ready()


def _init_cognitive_if_ready():
    """Initialize or upgrade cognitive layer when DB/API become available."""
    global _cognitive
    # Need at least DB pool for cognitive search
    if not _pg_pool:
        return
    # Initialize if missing, or upgrade if API client now available
    needs_init = not _cognitive or not _cognitive.available
    needs_upgrade = _cognitive and _api_client and not _cognitive._api
    if needs_init or needs_upgrade:
        try:
            _cognitive = CognitiveLayer(
                api_client=_api_client,  # May be None — search-only mode
                pg_pool=_pg_pool,
                embedding_pipeline=_embedding_pipeline,
            )
            mode = "full" if _api_client else "search-only"
            log.info(f"Cognitive layer initialized ({mode})")
        except Exception as exc:
            log.warning(f"Cognitive layer init failed: {exc}")


def _require_auth() -> NotebookLMAPIClient:
    global _api_client
    if _api_client is None:
        # Try auto-recovery: load from disk cache first
        cached = _load_cookies_from_disk()
        if cached:
            try:
                _api_client = NotebookLMAPIClient(cookies=cached)
                log.info("Auto-recovered auth from cached cookies")
                return _api_client
            except Exception:
                log.debug("Cached cookies expired, trying browser extraction")
        # Try auto-extract from Chrome
        try:
            from .cookie_extractor import auto_refresh

            cookie_string = auto_refresh()
            _api_client = NotebookLMAPIClient(cookies=cookie_string)
            log.info("Auto-recovered auth from Chrome cookies")
            return _api_client
        except Exception as e:
            log.debug(f"Chrome auto-extract failed: {e}")
        raise AuthenticationError(
            "Not authenticated. Either:\n"
            "1. Run `auto_auth` (extracts from Chrome automatically)\n"
            "2. Run `save_auth_tokens` with cookies from browser DevTools"
        )
    return _api_client


def _fmt(data: Any) -> str:
    if isinstance(data, str):
        return data
    return json.dumps(data, indent=2, default=str)


def _ok(msg: str) -> dict:
    return tool_result_content([text_content(msg)])


def _err(msg: str) -> dict:
    return tool_result_content([text_content(msg)], is_error=True)


# ══════════════════════════════════════════════════════════════════════════
# Tool Definitions
# ══════════════════════════════════════════════════════════════════════════

TOOLS = [
    # ── Notebooks ────────────────────────────────────────────────────────
    {
        "name": "notebook_list",
        "description": "List all NotebookLM notebooks. Returns IDs, titles, source counts, and timestamps.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "notebook_create",
        "description": "Create a new empty NotebookLM notebook.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Notebook title (optional, defaults to 'Untitled notebook')",
                },
            },
            "required": [],
        },
    },
    {
        "name": "notebook_get",
        "description": "Get notebook details including sources with types, statuses, and metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
            },
            "required": ["notebook_id"],
        },
    },
    {
        "name": "notebook_describe",
        "description": "Get AI-generated summary of notebook content with suggested topics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
            },
            "required": ["notebook_id"],
        },
    },
    {
        "name": "notebook_rename",
        "description": "Rename a notebook.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "new_title": {"type": "string", "description": "New title"},
            },
            "required": ["notebook_id", "new_title"],
        },
    },
    {
        "name": "notebook_delete",
        "description": "Delete a notebook. IRREVERSIBLE. Requires confirm=true.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to confirm deletion",
                },
            },
            "required": ["notebook_id", "confirm"],
        },
    },
    {
        "name": "chat_configure",
        "description": (
            "Configure chat behavior for a notebook. Set goal (default/custom/learning_guide), "
            "custom system prompt, and response length (default/longer/shorter)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "goal": {
                    "type": "string",
                    "enum": ["default", "custom", "learning_guide"],
                    "description": "Chat goal",
                },
                "custom_prompt": {
                    "type": "string",
                    "description": "Custom system prompt (required when goal='custom', max 10000 chars)",
                },
                "response_length": {
                    "type": "string",
                    "enum": ["default", "longer", "shorter"],
                    "description": "Response length preference",
                },
            },
            "required": ["notebook_id"],
        },
    },
    # ── Sources ──────────────────────────────────────────────────────────
    {
        "name": "source_add",
        "description": (
            "Add a source to a notebook. Supports: url (web page or YouTube), text (pasted content), "
            "drive (Google Doc/Sheets/Slides by document ID), file (upload PDF/TXT/MD/DOCX/CSV/MP3/MP4/images)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "source_type": {
                    "type": "string",
                    "enum": ["url", "text", "drive", "file"],
                    "description": "Source type",
                },
                "url": {
                    "type": "string",
                    "description": "URL for 'url' type (web page or YouTube)",
                },
                "text": {
                    "type": "string",
                    "description": "Text content for 'text' type",
                },
                "title": {
                    "type": "string",
                    "description": "Title for 'text' type (default: 'Pasted Text')",
                },
                "document_id": {
                    "type": "string",
                    "description": "Google Doc/Sheets/Slides ID for 'drive' type",
                },
                "drive_title": {
                    "type": "string",
                    "description": "Document title for 'drive' type",
                },
                "mime_type": {
                    "type": "string",
                    "description": "MIME type for 'drive' type (default: application/vnd.google-apps.document)",
                },
                "file_path": {
                    "type": "string",
                    "description": "Local file path for 'file' type",
                },
                "wait": {
                    "type": "boolean",
                    "description": "Wait for source processing to complete (default: false)",
                },
            },
            "required": ["notebook_id", "source_type"],
        },
    },
    {
        "name": "source_describe",
        "description": "Get AI-generated summary and keywords for a specific source.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Source UUID"},
            },
            "required": ["source_id"],
        },
    },
    {
        "name": "source_get_content",
        "description": "Get the raw text content of a source (no AI processing).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Source UUID"},
            },
            "required": ["source_id"],
        },
    },
    {
        "name": "source_list_drive",
        "description": "List all sources in a notebook with types, statuses, and Drive freshness info.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
            },
            "required": ["notebook_id"],
        },
    },
    {
        "name": "source_sync_drive",
        "description": "Sync a stale Google Drive source. Requires confirm=true.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Source UUID"},
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to confirm sync",
                },
            },
            "required": ["source_id", "confirm"],
        },
    },
    {
        "name": "source_delete",
        "description": "Delete a source from a notebook. IRREVERSIBLE. Requires confirm=true.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Source UUID"},
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to confirm deletion",
                },
            },
            "required": ["source_id", "confirm"],
        },
    },
    {
        "name": "source_rename",
        "description": (
            "Rename a source in place. Updates the display title without "
            "re-uploading or re-indexing content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "source_id": {"type": "string", "description": "Source UUID"},
                "new_title": {
                    "type": "string",
                    "description": "New display title (non-empty)",
                },
            },
            "required": ["notebook_id", "source_id", "new_title"],
        },
    },
    # ── Query / Conversation ─────────────────────────────────────────────
    {
        "name": "notebook_query",
        "description": (
            "Ask a question to a notebook. NotebookLM (Gemini) generates a response grounded in sources. "
            "Supports multi-turn conversations via conversation_id. "
            "Optionally limit to specific source_ids."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "query": {"type": "string", "description": "Your question"},
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of source IDs to scope the query",
                },
                "conversation_id": {
                    "type": "string",
                    "description": "Optional conversation ID for follow-up questions",
                },
            },
            "required": ["notebook_id", "query"],
        },
    },
    {
        "name": "notebook_query_start",
        "description": (
            "Kick off a notebook query in the background and return a job_id. "
            "Use for 50+ source notebooks where the synchronous notebook_query can "
            "exceed client timeouts. Poll with notebook_query_status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "query": {"type": "string", "description": "Your question"},
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of source IDs to scope the query",
                },
                "conversation_id": {
                    "type": "string",
                    "description": "Optional conversation ID for follow-up questions",
                },
                "timeout": {
                    "type": "number",
                    "description": "Per-job timeout in seconds (default 300)",
                },
            },
            "required": ["notebook_id", "query"],
        },
    },
    {
        "name": "notebook_query_status",
        "description": (
            "Check status of an async notebook query started via notebook_query_start. "
            "Returns status (pending|completed|failed|not_found), result (when complete), "
            "and elapsed_seconds. Finished jobs are reaped after 10 minutes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID returned from notebook_query_start",
                },
            },
            "required": ["job_id"],
        },
    },
    # ── Studio ───────────────────────────────────────────────────────────
    {
        "name": "studio_create",
        "description": (
            "Generate studio content from notebook sources. Types: audio (podcast-style overview), "
            "video (animated explainer), report (briefing doc/study guide/blog post), flashcards, quiz, "
            "infographic, slide_deck, data_table. Each type has specific options."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "type": {
                    "type": "string",
                    "enum": [
                        "audio",
                        "video",
                        "report",
                        "flashcards",
                        "quiz",
                        "infographic",
                        "slide_deck",
                        "data_table",
                    ],
                    "description": "Content type to generate",
                },
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional source IDs (default: all)",
                },
                "focus_prompt": {
                    "type": "string",
                    "description": "Optional focus/customization prompt",
                },
                "video_style_prompt": {
                    "type": "string",
                    "description": (
                        "Style prompt for video overviews when video_style='custom'. "
                        "Alias for focus_prompt — if both are set, this wins for type=video."
                    ),
                },
                "language": {
                    "type": "string",
                    "description": "Language code (default: en)",
                },
                "audio_format": {
                    "type": "string",
                    "enum": ["deep_dive", "brief", "critique", "debate"],
                    "description": "Audio format",
                },
                "audio_length": {
                    "type": "string",
                    "enum": ["short", "default", "long"],
                    "description": "Audio length",
                },
                "video_format": {
                    "type": "string",
                    "enum": ["explainer", "brief"],
                    "description": "Video format",
                },
                "video_style": {
                    "type": "string",
                    "enum": [
                        "auto_select",
                        "custom",
                        "classic",
                        "whiteboard",
                        "kawaii",
                        "anime",
                        "watercolor",
                        "retro_print",
                        "heritage",
                        "paper_craft",
                    ],
                    "description": "Video visual style",
                },
                "report_format": {
                    "type": "string",
                    "enum": [
                        "Briefing Doc",
                        "Study Guide",
                        "Blog Post",
                        "Create Your Own",
                    ],
                    "description": "Report format",
                },
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                    "description": "Flashcard/quiz difficulty",
                },
                "question_count": {
                    "type": "integer",
                    "description": "Number of quiz questions (default: 2)",
                },
                "orientation": {
                    "type": "string",
                    "enum": ["landscape", "portrait", "square"],
                    "description": "Infographic orientation",
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["concise", "standard", "detailed"],
                    "description": "Infographic detail level",
                },
                "slide_format": {
                    "type": "string",
                    "enum": ["detailed_deck", "presenter_slides"],
                    "description": "Slide deck format",
                },
                "slide_length": {
                    "type": "string",
                    "enum": ["short", "default"],
                    "description": "Slide deck length",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to start generation",
                },
            },
            "required": ["notebook_id", "type", "confirm"],
        },
    },
    {
        "name": "studio_status",
        "description": "Check status of all studio artifacts in a notebook.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
            },
            "required": ["notebook_id"],
        },
    },
    {
        "name": "studio_delete",
        "description": "Delete a studio artifact. IRREVERSIBLE. Requires confirm=true.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string", "description": "Artifact UUID"},
                "notebook_id": {
                    "type": "string",
                    "description": "Notebook UUID (optional, for mind map fallback)",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to confirm deletion",
                },
            },
            "required": ["artifact_id", "confirm"],
        },
    },
    {
        "name": "artifact_rename",
        "description": (
            "Rename a studio artifact (audio overview, report, video, mind map). "
            "Updates the display title without regenerating content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "artifact_id": {"type": "string", "description": "Artifact UUID"},
                "new_title": {
                    "type": "string",
                    "description": "New display title (non-empty)",
                },
            },
            "required": ["notebook_id", "artifact_id", "new_title"],
        },
    },
    {
        "name": "slide_deck_revise",
        "description": (
            "Revise a slide deck artifact with per-slide instructions. "
            "Creates a NEW artifact (does not modify the original). Slide-deck only — "
            "other studio types don't support revise. Poll with studio_status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "Existing slide deck artifact UUID to revise",
                },
                "notebook_id": {
                    "type": "string",
                    "description": "Notebook UUID (optional routing hint)",
                },
                "slide_instructions": {
                    "type": "array",
                    "description": "List of per-slide revision instructions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "0-based slide index",
                            },
                            "instruction": {
                                "type": "string",
                                "description": "Revision instruction for this slide",
                            },
                        },
                        "required": ["index", "instruction"],
                    },
                },
            },
            "required": ["artifact_id", "slide_instructions"],
        },
    },
    # ── Download / Export ────────────────────────────────────────────────
    {
        "name": "download_artifact",
        "description": "Download a studio artifact (audio MP3, video MP4, report PDF/MD, etc.). Returns download URL or content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "artifact_id": {"type": "string", "description": "Artifact UUID"},
            },
            "required": ["notebook_id", "artifact_id"],
        },
    },
    {
        "name": "export_artifact",
        "description": "Export a studio artifact to Google Docs or Sheets.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "artifact_id": {"type": "string", "description": "Artifact UUID"},
                "title": {
                    "type": "string",
                    "description": "Export title (default: 'NotebookLM Export')",
                },
                "export_type": {
                    "type": "string",
                    "enum": ["docs", "sheets"],
                    "description": "Export to Google Docs or Sheets",
                },
            },
            "required": ["notebook_id", "artifact_id"],
        },
    },
    # ── Research ─────────────────────────────────────────────────────────
    {
        "name": "research_start",
        "description": (
            "Start a research session to discover sources. "
            "Fast mode (~30s, ~10 sources) or deep mode (~5min, ~40 sources, web only)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "query": {"type": "string", "description": "Search query"},
                "source": {
                    "type": "string",
                    "enum": ["web", "drive"],
                    "description": "Search source (default: web)",
                },
                "mode": {
                    "type": "string",
                    "enum": ["fast", "deep"],
                    "description": "Research mode (default: fast)",
                },
            },
            "required": ["notebook_id", "query"],
        },
    },
    {
        "name": "research_status",
        "description": "Check research progress and get discovered sources. Poll until status='completed'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "task_id": {
                    "type": "string",
                    "description": "Optional task ID from research_start",
                },
            },
            "required": ["notebook_id"],
        },
    },
    {
        "name": "research_import",
        "description": "Import discovered research sources into the notebook.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "task_id": {"type": "string", "description": "Research task ID"},
                "source_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Indices of sources to import (from research_status results). If omitted, imports all.",
                },
            },
            "required": ["notebook_id", "task_id"],
        },
    },
    # ── Sharing ──────────────────────────────────────────────────────────
    {
        "name": "notebook_share_status",
        "description": "Get sharing settings and collaborators for a notebook.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
            },
            "required": ["notebook_id"],
        },
    },
    {
        "name": "notebook_share_public",
        "description": "Enable or disable public link access for a notebook.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "is_public": {
                    "type": "boolean",
                    "description": "True to enable public access, false to disable",
                },
            },
            "required": ["notebook_id", "is_public"],
        },
    },
    {
        "name": "notebook_share_invite",
        "description": "Invite a collaborator to a notebook by email.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "email": {"type": "string", "description": "Collaborator email"},
                "role": {
                    "type": "string",
                    "enum": ["editor", "viewer"],
                    "description": "Role (default: viewer)",
                },
                "notify": {
                    "type": "boolean",
                    "description": "Send email notification (default: true)",
                },
                "message": {
                    "type": "string",
                    "description": "Optional message in the invite email",
                },
            },
            "required": ["notebook_id", "email"],
        },
    },
    {
        "name": "notebook_share_batch",
        "description": (
            "Invite multiple collaborators to a notebook in one call. "
            "Loops through invites and reports per-email success/failure."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "invites": {
                    "type": "array",
                    "description": "List of {email, role} pairs",
                    "items": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "role": {
                                "type": "string",
                                "enum": ["editor", "viewer"],
                            },
                        },
                        "required": ["email"],
                    },
                },
                "notify": {
                    "type": "boolean",
                    "description": "Send email notification (default: true)",
                },
                "message": {
                    "type": "string",
                    "description": "Optional message applied to every invite",
                },
            },
            "required": ["notebook_id", "invites"],
        },
    },
    # ── Notes ────────────────────────────────────────────────────────────
    {
        "name": "note_create",
        "description": "Create a note in a notebook.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "content": {"type": "string", "description": "Note content"},
                "title": {
                    "type": "string",
                    "description": "Note title (default: 'New Note')",
                },
            },
            "required": ["notebook_id", "content"],
        },
    },
    {
        "name": "note_list",
        "description": "List all notes in a notebook.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
            },
            "required": ["notebook_id"],
        },
    },
    {
        "name": "note_update",
        "description": "Update a note's content or title.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "note_id": {"type": "string", "description": "Note UUID"},
                "content": {"type": "string", "description": "New content"},
                "title": {"type": "string", "description": "New title"},
            },
            "required": ["notebook_id", "note_id"],
        },
    },
    {
        "name": "note_delete",
        "description": "Delete a note. IRREVERSIBLE. Requires confirm=true.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "note_id": {"type": "string", "description": "Note UUID"},
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to confirm deletion",
                },
            },
            "required": ["notebook_id", "note_id", "confirm"],
        },
    },
    # ── Auth ──────────────────────────────────────────────────────────────
    {
        "name": "save_auth_tokens",
        "description": (
            "Save NotebookLM authentication cookies. Extract from Chrome DevTools: "
            "Network tab > any batchexecute request > Headers > Cookie header value."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "cookies": {
                    "type": "string",
                    "description": "Full cookie header string from Chrome DevTools",
                },
            },
            "required": ["cookies"],
        },
    },
    {
        "name": "refresh_auth",
        "description": "Refresh CSRF token and session ID by fetching the NotebookLM homepage.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "auto_auth",
        "description": (
            "Auto-extract NotebookLM cookies from Chrome's cookie database. "
            "No manual DevTools needed — reads encrypted cookies directly. "
            "First run requires macOS Keychain approval (click 'Always Allow'). "
            "Specify chrome_profile to use a non-default Chrome profile."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "chrome_profile": {
                    "type": "string",
                    "description": "Chrome profile directory name (default: 'Default'). Use 'Profile 1', 'Profile 2' etc. for additional profiles.",
                },
            },
            "required": [],
        },
    },
    # ── Cognitive Intelligence ───────────────────────────────────────────
    {
        "name": "cognitive_enrich_query",
        "description": (
            "GraphRAG-enriched query. Queries the notebook with context injected from your "
            "cognitive database: knowledge graph entities, cross-platform coherence moments, "
            "and FSRS-due insights. Returns enriched answer + enrichment details."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "query": {"type": "string", "description": "Your question"},
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional source IDs to scope",
                },
                "max_context_items": {
                    "type": "integer",
                    "description": "Max enrichment items (default: 5)",
                },
            },
            "required": ["notebook_id", "query"],
        },
    },
    {
        "name": "cognitive_search",
        "description": (
            "Hybrid search across the UCW cognitive database (140K+ events). "
            "Uses BM25 + semantic similarity with Reciprocal Rank Fusion. "
            "Useful for finding context to add to notebooks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "cognitive_insights",
        "description": (
            "Surface FSRS-due insights from the cognitive database. "
            "These are insights scheduled for spaced repetition review."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max insights (default: 10)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "research_to_notebook",
        "description": (
            "Bridge a ResearchGravity session into a NotebookLM notebook. "
            "Imports session URLs as URL sources, findings as pasted text, "
            "and configures chat with the session's thesis/gap."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "ResearchGravity session ID",
                },
                "notebook_id": {
                    "type": "string",
                    "description": "Optional existing notebook UUID (creates new if omitted)",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "knowledge_evolution",
        "description": (
            "Track how your understanding evolves over repeated queries to a notebook. "
            "Detects crystallization (mastery), fragmentation (needs exploration), or evolution."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string", "description": "Notebook UUID"},
                "query": {"type": "string", "description": "The query to track"},
                "response": {"type": "string", "description": "The response to record"},
            },
            "required": ["notebook_id", "query", "response"],
        },
    },
    {
        "name": "cognitive_auto_curate",
        "description": (
            "Check for coherence-driven notebook creation triggers. "
            "When high-significance cross-platform patterns are detected, "
            "optionally auto-create a notebook from convergent insights. "
            "Call without arc_id to see suggestions, with arc_id to create."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "arc_id": {
                    "type": "string",
                    "description": "Optional arc ID to auto-curate (from suggestions)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "cross_notebook_query",
        "description": (
            "Run the same query across multiple notebooks and aggregate the "
            "answers. If the cognitive layer is available, uses enriched_query "
            "for context-aware retrieval; otherwise falls back to plain "
            "notebook_query. Returns per-notebook answers with headers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of notebook UUIDs to query",
                    "minItems": 1,
                },
                "query": {"type": "string", "description": "Natural-language query"},
                "enriched": {
                    "type": "boolean",
                    "description": "Use cognitive enriched_query if available (default: true)",
                },
                "max_context_items": {
                    "type": "integer",
                    "description": "Enrichment context items per notebook (default: 5)",
                },
            },
            "required": ["notebook_ids", "query"],
        },
    },
    {
        "name": "batch_execute",
        "description": (
            "Execute multiple MCP tool calls in sequence within a single request. "
            "Each step is {tool, args}. Stops on first failure unless "
            "continue_on_error=true. Returns per-step status and output."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "Ordered list of {tool, args} to execute",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "args": {"type": "object"},
                        },
                        "required": ["tool"],
                    },
                    "minItems": 1,
                },
                "continue_on_error": {
                    "type": "boolean",
                    "description": "Keep running subsequent steps after a failure (default: false)",
                },
            },
            "required": ["steps"],
        },
    },
    {
        "name": "pipeline_research",
        "description": (
            "End-to-end research pipeline: create a notebook, add sources, "
            "run an optional first query, and optionally kick off a studio "
            "artifact (audio/report/video/etc). Returns a trace of each stage. "
            "Use this when you want one call to go from URLs to a ready notebook."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Notebook title"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URLs or text payloads to add as sources",
                },
                "source_type": {
                    "type": "string",
                    "enum": ["url", "text"],
                    "description": "How to treat items in `sources` (default: url)",
                },
                "query": {
                    "type": "string",
                    "description": "Optional first query to warm the conversation",
                },
                "studio": {
                    "type": "object",
                    "description": "Optional studio artifact to create after ingest",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "audio",
                                "video",
                                "report",
                                "flashcards",
                                "quiz",
                                "infographic",
                                "slide_deck",
                                "data_table",
                            ],
                        },
                        "focus_prompt": {"type": "string"},
                    },
                    "required": ["type"],
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to run studio creation side-effects",
                },
            },
            "required": ["title", "sources"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════
# Tool Handlers
# ══════════════════════════════════════════════════════════════════════════


async def handle_tool(name: str, args: dict) -> dict:
    """Route tool calls to handlers."""
    try:
        handler = _HANDLERS.get(name)
        if handler:
            return await handler(args)
        return _err(f"Unknown tool: {name}")
    except AuthenticationError as exc:
        return _err(f"Auth error: {exc}")
    except Exception as exc:
        log.error(f"Tool {name} error: {exc}", exc_info=True)
        return _err(f"Error: {exc}")


# ── Notebook Handlers ────────────────────────────────────────────────────


async def _h_notebook_list(args: dict) -> dict:
    client = _require_auth()
    notebooks = client.list_notebooks()
    if not notebooks:
        return _ok("No notebooks found.")
    lines = [f"Found {len(notebooks)} notebooks:\n"]
    for nb in notebooks:
        shared = " [shared]" if nb.is_shared else ""
        lines.append(f"  {nb.title}{shared}")
        lines.append(
            f"    ID: {nb.id} | Sources: {nb.source_count} | Modified: {nb.modified_at or 'unknown'}"
        )
    return _ok("\n".join(lines))


async def _h_notebook_create(args: dict) -> dict:
    client = _require_auth()
    nb = client.create_notebook(args.get("title", ""))
    if nb:
        return _ok(f"Created notebook: {nb.title}\n  ID: {nb.id}\n  URL: {nb.url}")
    return _err("Failed to create notebook.")


async def _h_notebook_get(args: dict) -> dict:
    client = _require_auth()
    nid = args["notebook_id"]
    sources = client.get_notebook_sources_with_types(nid)
    lines = [f"Notebook: {nid}\nSources ({len(sources)}):\n"]
    for s in sources:
        status = (
            "ready"
            if s.get("status") == 2
            else "processing"
            if s.get("status") == 1
            else "error"
            if s.get("status") == 3
            else "unknown"
        )
        lines.append(f"  {s['title']} [{s['source_type_name']}] ({status})")
        lines.append(
            f"    ID: {s['id']}" + (f" | URL: {s['url']}" if s.get("url") else "")
        )
    return _ok("\n".join(lines))


async def _h_notebook_describe(args: dict) -> dict:
    client = _require_auth()
    result = client.get_notebook_summary(args["notebook_id"])
    lines = [f"Summary:\n{result['summary']}\n"]
    if result.get("suggested_topics"):
        lines.append("Suggested Topics:")
        for t in result["suggested_topics"]:
            lines.append(f"  - {t['question']}")
    return _ok("\n".join(lines))


async def _h_notebook_rename(args: dict) -> dict:
    client = _require_auth()
    ok = client.rename_notebook(args["notebook_id"], args["new_title"])
    return (
        _ok(f"Renamed to: {args['new_title']}")
        if ok
        else _err("Failed to rename notebook.")
    )


async def _h_notebook_delete(args: dict) -> dict:
    if not args.get("confirm"):
        return _err("Deletion requires confirm=true. This is IRREVERSIBLE.")
    client = _require_auth()
    ok = client.delete_notebook(args["notebook_id"])
    return (
        _ok(f"Deleted notebook {args['notebook_id']}")
        if ok
        else _err("Failed to delete notebook.")
    )


async def _h_chat_configure(args: dict) -> dict:
    client = _require_auth()
    result = client.configure_chat(
        args["notebook_id"],
        goal=args.get("goal", "default"),
        custom_prompt=args.get("custom_prompt"),
        response_length=args.get("response_length", "default"),
    )
    if result.get("status") == "success":
        return _ok(
            f"Chat configured: goal={result['goal']}, length={result['response_length']}"
        )
    return _err(result.get("error", "Failed to configure chat."))


# ── Source Handlers ──────────────────────────────────────────────────────


async def _h_source_add(args: dict) -> dict:
    client = _require_auth()
    nid = args["notebook_id"]
    stype = args["source_type"]
    wait = args.get("wait", False)

    if stype == "url":
        url = args.get("url")
        if not url:
            return _err("'url' is required for source_type='url'")
        result = client.add_url_source(nid, url, wait=wait)
    elif stype == "text":
        text = args.get("text")
        if not text:
            return _err("'text' is required for source_type='text'")
        result = client.add_text_source(
            nid, text, title=args.get("title", "Pasted Text"), wait=wait
        )
    elif stype == "drive":
        doc_id = args.get("document_id")
        if not doc_id:
            return _err("'document_id' is required for source_type='drive'")
        result = client.add_drive_source(
            nid,
            doc_id,
            title=args.get("drive_title", "Drive Document"),
            mime_type=args.get("mime_type", "application/vnd.google-apps.document"),
            wait=wait,
        )
    elif stype == "file":
        fp = args.get("file_path")
        if not fp:
            return _err("'file_path' is required for source_type='file'")
        result = client.add_file(nid, fp, wait=wait)
    else:
        return _err(f"Invalid source_type: {stype}")

    if result:
        return _ok(
            f"Source added: {result.get('title', 'Unknown')}\n  ID: {result.get('id')}"
        )
    return _err("Failed to add source.")


async def _h_source_describe(args: dict) -> dict:
    client = _require_auth()
    result = client.get_source_guide(args["source_id"])
    lines = [f"Summary:\n{result['summary']}"]
    if result.get("keywords"):
        lines.append(f"\nKeywords: {', '.join(result['keywords'])}")
    return _ok("\n".join(lines))


async def _h_source_get_content(args: dict) -> dict:
    client = _require_auth()
    result = client.get_source_fulltext(args["source_id"])
    lines = [
        f"Title: {result['title']}",
        f"Type: {result['source_type']}",
        f"Characters: {result['char_count']}",
    ]
    if result.get("url"):
        lines.append(f"URL: {result['url']}")
    lines.append(f"\n{result['content'][:5000]}")
    if result["char_count"] > 5000:
        lines.append(
            f"\n... (truncated, {result['char_count'] - 5000} chars remaining)"
        )
    return _ok("\n".join(lines))


async def _h_source_list_drive(args: dict) -> dict:
    client = _require_auth()
    sources = client.get_notebook_sources_with_types(args["notebook_id"])
    lines = [f"Sources ({len(sources)}):\n"]
    for s in sources:
        status = (
            "ready"
            if s.get("status") == 2
            else "processing"
            if s.get("status") == 1
            else "error"
            if s.get("status") == 3
            else "unknown"
        )
        sync_info = ""
        if s.get("can_sync"):
            fresh = client.check_source_freshness(s["id"])
            sync_info = (
                " [STALE - needs sync]"
                if fresh is False
                else " [fresh]"
                if fresh
                else ""
            )
        lines.append(f"  {s['title']} [{s['source_type_name']}] ({status}){sync_info}")
        lines.append(f"    ID: {s['id']}")
    return _ok("\n".join(lines))


async def _h_source_sync_drive(args: dict) -> dict:
    if not args.get("confirm"):
        return _err("Sync requires confirm=true.")
    client = _require_auth()
    result = client.sync_drive_source(args["source_id"])
    if result:
        return _ok(f"Synced: {result.get('title', 'Unknown')} (ID: {result.get('id')})")
    return _err("Failed to sync source.")


async def _h_source_delete(args: dict) -> dict:
    if not args.get("confirm"):
        return _err("Deletion requires confirm=true. This is IRREVERSIBLE.")
    client = _require_auth()
    ok = client.delete_source(args["source_id"])
    return (
        _ok(f"Deleted source {args['source_id']}")
        if ok
        else _err("Failed to delete source.")
    )


async def _h_source_rename(args: dict) -> dict:
    client = _require_auth()
    new_title = (args.get("new_title") or "").strip()
    if not new_title:
        return _err("new_title must be non-empty.")
    result = client.rename_source(
        args["notebook_id"], args["source_id"], new_title
    )
    if not result:
        return _err("Failed to rename source.")
    return _ok(
        f"Renamed source {result.get('id', args['source_id'])} → {result.get('title', new_title)!r}"
    )


# ── Query Handler ────────────────────────────────────────────────────────


async def _h_notebook_query(args: dict) -> dict:
    client = _require_auth()
    result = client.query(
        args["notebook_id"],
        args["query"],
        source_ids=args.get("source_ids"),
        conversation_id=args.get("conversation_id"),
    )
    if not result or not result.get("answer"):
        return _err("No response from NotebookLM.")
    lines = [
        result["answer"],
        f"\n---\nConversation: {result['conversation_id']} | Turn: {result['turn_number']}",
    ]
    if result.get("is_follow_up"):
        lines.append("(follow-up)")
    # Capture as cognitive event if available
    if _cognitive and _cognitive.available:
        await _cognitive.capture_result(
            "notebook_query",
            {
                "query": args["query"],
                "answer": result["answer"][:500],
                "notebook_id": args["notebook_id"],
            },
        )
    return _ok("\n".join(lines))


async def _h_notebook_query_start(args: dict) -> dict:
    client = _require_auth()
    job_id = client.query_start(
        notebook_id=args["notebook_id"],
        query_text=args["query"],
        source_ids=args.get("source_ids"),
        conversation_id=args.get("conversation_id"),
        timeout=float(args.get("timeout", 300.0)),
    )
    return _ok(
        f"Query job started: {job_id}\n"
        f"Poll with notebook_query_status(job_id='{job_id}')."
    )


async def _h_notebook_query_status(args: dict) -> dict:
    client = _require_auth()
    snap = client.query_status(args["job_id"])
    status = snap.get("status")
    if status == "not_found":
        return _err(f"Job {args['job_id']} not found (may have been reaped after 10 min).")
    lines = [
        f"Job: {snap.get('job_id')}",
        f"Status: {status}",
        f"Elapsed: {snap.get('elapsed_seconds')}s",
    ]
    if status == "failed":
        lines.append(f"Error: {snap.get('error')}")
        return _err("\n".join(lines))
    if status == "completed":
        result = snap.get("result") or {}
        answer = result.get("answer") or ""
        lines.append("")
        lines.append(answer)
        lines.append(
            f"\n---\nConversation: {result.get('conversation_id')} | "
            f"Turn: {result.get('turn_number')}"
        )
        if _cognitive and _cognitive.available and answer:
            await _cognitive.capture_result(
                "notebook_query",
                {
                    "query": snap.get("query"),
                    "answer": answer[:500],
                    "notebook_id": snap.get("notebook_id"),
                },
            )
    return _ok("\n".join(lines))


# ── Studio Handlers ──────────────────────────────────────────────────────


async def _h_studio_create(args: dict) -> dict:
    if not args.get("confirm"):
        return _err("Studio creation requires confirm=true.")
    client = _require_auth()
    nid = args["notebook_id"]
    stype = args["type"]
    sids = args.get("source_ids")
    lang = args.get("language", "en")
    focus = args.get("focus_prompt", "")

    if stype == "audio":
        fc = constants.AUDIO_FORMATS.get_code(args.get("audio_format", "deep_dive"))
        lc = constants.AUDIO_LENGTHS.get_code(args.get("audio_length", "default"))
        result = client.create_audio_overview(
            nid, sids, format_code=fc, length_code=lc, language=lang, focus_prompt=focus
        )
    elif stype == "video":
        fc = constants.VIDEO_FORMATS.get_code(args.get("video_format", "explainer"))
        vstyle = args.get("video_style", "auto_select")
        vs = constants.VIDEO_STYLES.get_code(vstyle)
        video_prompt = (args.get("video_style_prompt") or focus or "").strip()
        if vstyle == "custom" and not video_prompt:
            log.warning(
                "studio_create: video_style=custom with no video_style_prompt/focus_prompt — result may be generic."
            )
        result = client.create_video_overview(
            nid,
            sids,
            format_code=fc,
            visual_style_code=vs,
            language=lang,
            focus_prompt=video_prompt,
        )
    elif stype == "report":
        rf = args.get("report_format", "Briefing Doc")
        result = client.create_report(
            nid, sids, report_format=rf, custom_prompt=focus, language=lang
        )
    elif stype == "flashcards":
        dc = constants.FLASHCARD_DIFFICULTIES.get_code(args.get("difficulty", "medium"))
        result = client.create_flashcards(nid, sids, difficulty_code=dc)
    elif stype == "quiz":
        dc = constants.FLASHCARD_DIFFICULTIES.get_code(args.get("difficulty", "medium"))
        qc = args.get("question_count", 2)
        result = client.create_quiz(nid, sids, question_count=qc, difficulty=dc)
    elif stype == "infographic":
        oc = constants.INFOGRAPHIC_ORIENTATIONS.get_code(
            args.get("orientation", "landscape")
        )
        dl = constants.INFOGRAPHIC_DETAILS.get_code(
            args.get("detail_level", "standard")
        )
        result = client.create_infographic(
            nid,
            sids,
            orientation_code=oc,
            detail_level_code=dl,
            language=lang,
            focus_prompt=focus,
        )
    elif stype == "slide_deck":
        sf = constants.SLIDE_DECK_FORMATS.get_code(
            args.get("slide_format", "detailed_deck")
        )
        sl = constants.SLIDE_DECK_LENGTHS.get_code(args.get("slide_length", "default"))
        result = client.create_slide_deck(
            nid, sids, format_code=sf, length_code=sl, language=lang, focus_prompt=focus
        )
    elif stype == "data_table":
        result = client.create_data_table(nid, sids, description=focus, language=lang)
    else:
        return _err(f"Unknown studio type: {stype}")

    if result:
        return _ok(
            f"Studio '{stype}' creation started.\n  Artifact ID: {result['artifact_id']}\n  Status: {result['status']}"
        )
    return _err(f"Failed to create {stype}.")


async def _h_studio_status(args: dict) -> dict:
    client = _require_auth()
    artifacts = client.poll_studio_status(args["notebook_id"])
    if not artifacts:
        return _ok("No studio artifacts found.")
    lines = [f"Studio Artifacts ({len(artifacts)}):\n"]
    for a in artifacts:
        lines.append(f"  {a['title']} [{a['type']}] - {a['status']}")
        lines.append(f"    ID: {a['artifact_id']}")
    return _ok("\n".join(lines))


async def _h_studio_delete(args: dict) -> dict:
    if not args.get("confirm"):
        return _err("Deletion requires confirm=true. This is IRREVERSIBLE.")
    client = _require_auth()
    ok = client.delete_studio_artifact(
        args["artifact_id"], notebook_id=args.get("notebook_id")
    )
    return (
        _ok(f"Deleted artifact {args['artifact_id']}")
        if ok
        else _err("Failed to delete artifact.")
    )


async def _h_artifact_rename(args: dict) -> dict:
    client = _require_auth()
    new_title = (args.get("new_title") or "").strip()
    if not new_title:
        return _err("new_title must be non-empty.")
    try:
        ok = client.rename_artifact(
            args["notebook_id"], args["artifact_id"], new_title
        )
    except ValueError as e:
        return _err(str(e))
    return (
        _ok(f"Renamed artifact {args['artifact_id']} → {new_title!r}")
        if ok
        else _err("Failed to rename artifact.")
    )


async def _h_slide_deck_revise(args: dict) -> dict:
    client = _require_auth()
    instructions_raw = args.get("slide_instructions") or []
    if not instructions_raw:
        return _err("slide_instructions must not be empty.")
    pairs: list[tuple[int, str]] = []
    for item in instructions_raw:
        if not isinstance(item, dict):
            return _err("Each slide_instruction must be an object.")
        idx = item.get("index")
        text = (item.get("instruction") or "").strip()
        if idx is None or not isinstance(idx, int):
            return _err("Each slide_instruction needs an integer index.")
        if not text:
            return _err("Each slide_instruction needs non-empty instruction text.")
        pairs.append((idx, text))
    try:
        result = client.revise_slide_deck(
            args["artifact_id"], pairs, notebook_id=args.get("notebook_id")
        )
    except ValueError as e:
        return _err(str(e))
    if not result or not result.get("artifact_id"):
        return _err("Slide deck revise returned no new artifact.")
    return _ok(
        f"Revised slide deck → new artifact {result['artifact_id']} "
        f"({result.get('status', 'unknown')}) from {result.get('original_artifact_id')}"
    )


# ── Download / Export Handlers ───────────────────────────────────────────


async def _h_download_artifact(args: dict) -> dict:
    return _err(
        "download_artifact is not yet supported via HTTP/RPC. "
        "Use export_artifact to export to Google Docs/Sheets, "
        "or access the content via the NotebookLM web UI."
    )


async def _h_export_artifact(args: dict) -> dict:
    client = _require_auth()
    result = client.export_artifact(
        args["notebook_id"],
        args["artifact_id"],
        title=args.get("title", "NotebookLM Export"),
        export_type=args.get("export_type", "docs"),
    )
    if result.get("status") == "success":
        return _ok(f"Exported: {result['url']}")
    return _err(result.get("message", "Export failed."))


# ── Research Handlers ────────────────────────────────────────────────────


async def _h_research_start(args: dict) -> dict:
    client = _require_auth()
    result = client.start_research(
        args["notebook_id"],
        args["query"],
        source=args.get("source", "web"),
        mode=args.get("mode", "fast"),
    )
    if result:
        return _ok(
            f"Research started ({result['mode']} mode, {result['source']} source)\n"
            f"  Task ID: {result['task_id']}\n"
            f"  Query: {result['query']}\n"
            f"Poll with research_status to check progress."
        )
    return _err("Failed to start research.")


async def _h_research_status(args: dict) -> dict:
    client = _require_auth()
    result = client.poll_research(
        args["notebook_id"], target_task_id=args.get("task_id")
    )
    if not result or result.get("status") == "no_research":
        return _ok("No active research found.")
    lines = [
        f"Research: {result.get('query', '')}",
        f"Status: {result['status']} | Mode: {result.get('mode', '?')} | Source: {result.get('source_type', '?')}",
        f"Sources found: {result.get('source_count', 0)}",
    ]
    if result.get("summary"):
        lines.append(f"\nSummary: {result['summary'][:500]}")
    if result.get("sources"):
        lines.append("\nDiscovered Sources:")
        for s in result["sources"]:
            lines.append(
                f"  [{s['index']}] {s['title']}"
                + (f" - {s['url']}" if s.get("url") else "")
            )
    return _ok("\n".join(lines))


async def _h_research_import(args: dict) -> dict:
    client = _require_auth()
    nid = args["notebook_id"]
    tid = args["task_id"]

    # First get current research results to select sources
    status = client.poll_research(nid, target_task_id=tid)
    if not status or not status.get("sources"):
        return _err("No sources found in research results.")

    sources = status["sources"]
    indices = args.get("source_indices")
    if indices:
        sources = [s for s in sources if s.get("index") in indices]

    imported = client.import_research_sources(nid, tid, sources)
    if imported:
        lines = [f"Imported {len(imported)} sources:"]
        for s in imported:
            lines.append(f"  {s['title']} (ID: {s['id']})")
        return _ok("\n".join(lines))
    return _err("No sources were imported.")


# ── Sharing Handlers ─────────────────────────────────────────────────────


async def _h_notebook_share_status(args: dict) -> dict:
    client = _require_auth()
    status = client.get_share_status(args["notebook_id"])
    lines = [
        f"Access: {status.access_level}",
    ]
    if status.public_link:
        lines.append(f"Public Link: {status.public_link}")
    if status.collaborators:
        lines.append(f"\nCollaborators ({len(status.collaborators)}):")
        for c in status.collaborators:
            pending = " (pending)" if c.is_pending else ""
            name = f" ({c.display_name})" if c.display_name else ""
            lines.append(f"  {c.email}{name} - {c.role}{pending}")
    else:
        lines.append("No collaborators.")
    return _ok("\n".join(lines))


async def _h_notebook_share_public(args: dict) -> dict:
    client = _require_auth()
    link = client.set_public_access(args["notebook_id"], is_public=args["is_public"])
    if args["is_public"]:
        return _ok(f"Public access enabled.\nLink: {link}")
    return _ok("Public access disabled.")


async def _h_notebook_share_invite(args: dict) -> dict:
    client = _require_auth()
    ok = client.add_collaborator(
        args["notebook_id"],
        args["email"],
        role=args.get("role", "viewer"),
        notify=args.get("notify", True),
        message=args.get("message", ""),
    )
    return (
        _ok(f"Invited {args['email']} as {args.get('role', 'viewer')}")
        if ok
        else _err("Failed to invite collaborator.")
    )


async def _h_notebook_share_batch(args: dict) -> dict:
    client = _require_auth()
    invites = args.get("invites") or []
    if not invites:
        return _err("invites must be a non-empty list of {email, role}.")
    notify = args.get("notify", True)
    message = args.get("message", "")
    nid = args["notebook_id"]
    succeeded: list[str] = []
    failed: list[str] = []
    for entry in invites:
        email = (entry.get("email") or "").strip()
        if not email:
            failed.append("<missing email>")
            continue
        role = entry.get("role", "viewer")
        try:
            ok = client.add_collaborator(
                nid, email, role=role, notify=notify, message=message
            )
        except Exception as e:
            log.warning(f"share_batch: {email} failed: {e}")
            ok = False
        (succeeded if ok else failed).append(f"{email} ({role})")
    lines = [
        f"Share batch complete: {len(succeeded)}/{len(invites)} succeeded."
    ]
    if succeeded:
        lines.append("Invited:")
        lines.extend(f"  {x}" for x in succeeded)
    if failed:
        lines.append("Failed:")
        lines.extend(f"  {x}" for x in failed)
    return _ok("\n".join(lines)) if succeeded else _err("\n".join(lines))


# ── Note Handlers ────────────────────────────────────────────────────────


async def _h_note_create(args: dict) -> dict:
    client = _require_auth()
    result = client.create_note(
        args["notebook_id"], args["content"], title=args.get("title")
    )
    if result:
        return _ok(f"Note created: {result['title']}\n  ID: {result['id']}")
    return _err("Failed to create note.")


async def _h_note_list(args: dict) -> dict:
    client = _require_auth()
    notes = client.list_notes(args["notebook_id"])
    if not notes:
        return _ok("No notes found.")
    lines = [f"Notes ({len(notes)}):\n"]
    for n in notes:
        lines.append(f"  {n['title']}")
        lines.append(f"    ID: {n['id']} | Preview: {n['preview']}")
    return _ok("\n".join(lines))


async def _h_note_update(args: dict) -> dict:
    client = _require_auth()
    result = client.update_note(
        args["note_id"],
        content=args.get("content"),
        title=args.get("title"),
        notebook_id=args["notebook_id"],
    )
    if result:
        return _ok(f"Updated note: {result['title']}")
    return _err("Failed to update note.")


async def _h_note_delete(args: dict) -> dict:
    if not args.get("confirm"):
        return _err("Deletion requires confirm=true. This is IRREVERSIBLE.")
    client = _require_auth()
    ok = client.delete_note(args["note_id"], args["notebook_id"])
    return (
        _ok(f"Deleted note {args['note_id']}") if ok else _err("Failed to delete note.")
    )


# ── Auth Handlers ────────────────────────────────────────────────────────


def _get_cookie_cache_path() -> Path:
    """Get persistent cookie cache file path."""
    from ..config_notebooklm import NotebookLMConfig

    return NotebookLMConfig.AUTH_STATE_DIR / "cookies.txt"


def _save_cookies_to_disk(cookies: str) -> None:
    """Persist cookies to disk for cross-session reuse."""
    cache_path = _get_cookie_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(cookies)
    cache_path.chmod(0o600)  # Owner-only read/write
    log.info(f"Cookies persisted to {cache_path}")


def _load_cookies_from_disk() -> str:
    """Load persisted cookies from disk. Returns empty string if none."""
    cache_path = _get_cookie_cache_path()
    if cache_path.exists():
        cookies = cache_path.read_text().strip()
        if cookies:
            log.info(f"Loaded cached cookies from {cache_path}")
            return cookies
    return ""


async def _h_save_auth_tokens(args: dict) -> dict:
    import os
    import asyncio

    global _api_client, _pg_pool
    cookies = args["cookies"]
    # Save to env for current session
    os.environ["NOTEBOOKLM_COOKIES"] = cookies
    # Persist to disk for future sessions
    _save_cookies_to_disk(cookies)
    try:
        _api_client = NotebookLMAPIClient(cookies=cookies)
        # Verify by listing notebooks
        nbs = _api_client.list_notebooks()

        # Wait for DB background init if not ready yet (up to 5s)
        if _pg_pool is None:
            for _ in range(10):
                await asyncio.sleep(0.5)
                if _pg_pool is not None:
                    break

        # Re-initialize cognitive layer now that we have auth + DB may be ready
        _init_cognitive_if_ready()
        cognitive_status = (
            "cognitive=active"
            if (_cognitive and _cognitive.available)
            else "cognitive=pending"
        )
        return _ok(
            f"Authenticated. Found {len(nbs)} notebooks. Cookies persisted to disk. ({cognitive_status})"
        )
    except Exception as exc:
        _api_client = None
        return _err(f"Authentication failed: {exc}")


async def _h_refresh_auth(args: dict) -> dict:
    client = _require_auth()
    try:
        client._refresh_auth_tokens()
        client._client = None  # Force new HTTP client
        return _ok("Auth tokens refreshed.")
    except Exception as exc:
        return _err(f"Refresh failed: {exc}")


async def _h_auto_auth(args: dict) -> dict:
    """Auto-extract cookies from Chrome and authenticate."""
    import os
    import asyncio

    global _api_client, _pg_pool

    chrome_profile = args.get("chrome_profile", "Default")

    try:
        import notebooklm_mcp.tools.cookie_extractor as ce
        from .cookie_extractor import cookies_to_header, save_cookies

        # Override DB path if non-default profile
        if chrome_profile != "Default":
            ce.CHROME_COOKIE_DB = (
                Path.home()
                / "Library/Application Support/Google/Chrome"
                / chrome_profile
                / "Cookies"
            )

        cookies = ce.extract_chrome_cookies()
        cookie_string = cookies_to_header(cookies)
        save_cookies(cookie_string)
        os.environ["NOTEBOOKLM_COOKIES"] = cookie_string

        _api_client = NotebookLMAPIClient(cookies=cookie_string)
        nbs = _api_client.list_notebooks()

        # Wait for DB background init if not ready yet (up to 5s)
        if _pg_pool is None:
            for _ in range(10):
                await asyncio.sleep(0.5)
                if _pg_pool is not None:
                    break

        _init_cognitive_if_ready()
        cognitive_status = (
            "cognitive=active"
            if (_cognitive and _cognitive.available)
            else "cognitive=pending"
        )

        return _ok(
            f"Auto-authenticated from Chrome ({chrome_profile}). "
            f"Extracted {len(cookies)} cookies. "
            f"Found {len(nbs)} notebooks. ({cognitive_status})"
        )
    except PermissionError as e:
        return _err(str(e))
    except FileNotFoundError as e:
        return _err(f"Chrome cookie DB not found: {e}")
    except Exception as exc:
        _api_client = None
        return _err(f"Auto-auth failed: {exc}")


# ── Cognitive Handlers ───────────────────────────────────────────────────


async def _h_cognitive_enrich_query(args: dict) -> dict:
    _require_auth()
    if not _cognitive or not _cognitive.available:
        return _err(
            "Cognitive layer not available. PostgreSQL + coherence_engine required."
        )
    result = await _cognitive.enriched_query(
        args["notebook_id"],
        args["query"],
        source_ids=args.get("source_ids"),
        max_context_items=args.get("max_context_items", 5),
    )
    if not result or not result.get("answer"):
        return _err("No response from enriched query.")
    lines = [result["answer"]]
    if result.get("enrichments_used", 0) > 0:
        lines.append(f"\n---\nEnrichments applied: {result['enrichments_used']}")
        for detail in result.get("enrichment_details", []):
            lines.append(f"\n{detail[:200]}")
    lines.append(f"Conversation: {result.get('conversation_id', 'N/A')}")
    return _ok("\n".join(lines))


async def _h_cognitive_search(args: dict) -> dict:
    if not _cognitive or not _cognitive.available:
        return _err("Cognitive layer not available.")
    results = await _cognitive.cognitive_search(
        args["query"], limit=args.get("limit", 10)
    )
    if not results:
        return _ok("No results found.")
    lines = [f"Found {len(results)} results:\n"]
    for r in results:
        # Handle both dict and HybridResult objects
        if isinstance(r, dict):
            score = r.get("score", r.get("rrf_score", 0)) or 0
            platform = r.get("platform", "unknown")
            content = r.get("content", r.get("preview", ""))[:200]
            session = r.get("session_id", "")[:16]
            mode = r.get("cognitive_mode", "")
        else:
            score = getattr(r, "rrf_score", 0) or 0
            platform = getattr(r, "platform", "unknown")
            content = getattr(r, "preview", "")[:200]
            session = (getattr(r, "session_id", "") or "")[:16]
            mode = getattr(r, "cognitive_mode", "")
        lines.append(f"  [{score:.4f}] [{platform}] [{mode}] {content}")
    return _ok("\n".join(lines))


async def _h_cognitive_insights(args: dict) -> dict:
    if not _cognitive or not _cognitive.available:
        return _err("Cognitive layer not available.")
    insights = await _cognitive.get_due_insights(limit=args.get("limit", 10))
    if not insights:
        return _ok("No insights due for review.")
    lines = [f"Due for review ({len(insights)}):\n"]
    for i in insights:
        desc = i.get("description", i.get("content", ""))
        platforms = i.get("platforms", [])
        ctype = i.get("coherence_type", "")
        plat_str = f" [{', '.join(platforms)}]" if platforms else ""
        lines.append(f"  - [{ctype}]{plat_str} {desc[:200]}")
    return _ok("\n".join(lines))


async def _h_research_to_notebook(args: dict) -> dict:
    _require_auth()
    if not _cognitive or not _cognitive.available:
        return _err("Cognitive layer not available.")
    result = await _cognitive.import_research_session(
        args["session_id"],
        notebook_id=args.get("notebook_id"),
    )
    if result:
        return _ok(
            f"Session imported into notebook.\n"
            f"  Notebook: {result['notebook_id']}\n"
            f"  Topic: {result['topic']}\n"
            f"  Sources added: {result['sources_added']} ({result['url_count']} URLs, {result['finding_count']} findings)"
        )
    return _err("Failed to import research session.")


async def _h_knowledge_evolution(args: dict) -> dict:
    _require_auth()
    if not _cognitive or not _cognitive.available:
        return _err("Cognitive layer not available.")
    result = await _cognitive.track_knowledge_evolution(
        args["notebook_id"],
        args["query"],
        args["response"],
    )
    if result.get("tracked"):
        lines = [
            "Knowledge tracked.",
            f"  State: {result.get('evolution_state', 'unknown')}",
            f"  Data points: {result.get('data_points', 0)}",
        ]
        if "normalized_variance" in result:
            lines.append(f"  Variance: {result['normalized_variance']}")
        return _ok("\n".join(lines))
    return _err(result.get("error", "Failed to track knowledge evolution."))


async def _h_cognitive_auto_curate(args: dict) -> dict:
    if not _cognitive or not _cognitive.available:
        return _err("Cognitive layer not available.")
    arc_id = args.get("arc_id")
    if arc_id:
        _require_auth()
        result = await _cognitive.auto_curate_notebook(arc_id)
        if result:
            return _ok(
                f"Auto-curated notebook created.\n"
                f"  Notebook: {result['notebook_id']} ({result['title']})\n"
                f"  Topic: {result['topic']}\n"
                f"  Moments included: {result['moments_included']}"
            )
        return _err("Failed to auto-curate notebook.")
    else:
        suggestions = await _cognitive.check_curation_triggers()
        if not suggestions:
            return _ok("No curation triggers detected.")
        lines = ["Curation suggestions:\n"]
        for s in suggestions:
            lines.append(f"  {s['suggested_title']}")
            lines.append(
                f"    Arc ID: {s['arc_id']} | Significance: {s['significance']:.2f} | Platforms: {s['platforms']} | Moments: {s['moments']}"
            )
        lines.append("\nCall cognitive_auto_curate with arc_id to create a notebook.")
        return _ok("\n".join(lines))


# ── Phase 3: Convergence Handlers ────────────────────────────────────────


async def _h_cross_notebook_query(args: dict) -> dict:
    client = _require_auth()
    notebook_ids = args.get("notebook_ids") or []
    query = (args.get("query") or "").strip()
    if not notebook_ids:
        return _err("notebook_ids must be a non-empty list.")
    if not query:
        return _err("query must be non-empty.")
    use_enriched = args.get("enriched", True) and _cognitive and _cognitive.available
    max_ctx = args.get("max_context_items", 5)

    sections: list[str] = [f"Cross-notebook query: {query}"]
    sections.append(f"Mode: {'enriched' if use_enriched else 'plain'}")
    sections.append(f"Notebooks: {len(notebook_ids)}\n")

    for nid in notebook_ids:
        header = f"── {nid} ──"
        try:
            if use_enriched:
                result = await _cognitive.enriched_query(
                    nid, query, max_context_items=max_ctx
                )
                answer = (result or {}).get("answer") or ""
            else:
                result = client.query(nid, query)
                answer = (result or {}).get("answer") or ""
        except Exception as e:
            log.warning(f"cross_notebook_query {nid}: {e}")
            answer = f"[error: {e}]"
        if not answer:
            answer = "[no answer]"
        sections.append(f"{header}\n{answer}\n")

    if _cognitive and _cognitive.available:
        await _cognitive.capture_result(
            "cross_notebook_query",
            {"query": query, "notebook_count": len(notebook_ids)},
        )
    return _ok("\n".join(sections))


async def _h_batch_execute(args: dict) -> dict:
    steps = args.get("steps") or []
    if not steps:
        return _err("steps must be a non-empty list.")
    continue_on_error = args.get("continue_on_error", False)
    results: list[dict] = []
    lines: list[str] = [f"Batch: {len(steps)} steps"]
    ok_count = 0
    for i, step in enumerate(steps, 1):
        tool = step.get("tool")
        step_args = step.get("args") or {}
        if not tool:
            lines.append(f"  [{i}] ERROR: missing 'tool'")
            results.append({"step": i, "ok": False, "error": "missing tool"})
            if not continue_on_error:
                break
            continue
        if tool == "batch_execute":
            lines.append(f"  [{i}] ERROR: nested batch_execute not allowed")
            results.append({"step": i, "ok": False, "error": "nested batch"})
            if not continue_on_error:
                break
            continue
        try:
            result = await handle_tool(tool, step_args)
        except Exception as e:
            result = _err(f"{type(e).__name__}: {e}")
        is_err = bool(result.get("isError"))
        results.append({"step": i, "tool": tool, "ok": not is_err, "result": result})
        marker = "FAIL" if is_err else "OK"
        if not is_err:
            ok_count += 1
        preview = ""
        try:
            content = result.get("content") or []
            if content and isinstance(content, list):
                preview = (content[0].get("text") or "")[:120]
        except Exception:
            preview = ""
        lines.append(f"  [{i}] {marker} {tool}: {preview}")
        if is_err and not continue_on_error:
            lines.append(f"  Stopped at step {i} (continue_on_error=false)")
            break
    lines.insert(1, f"Completed: {ok_count}/{len(steps)} ok")
    return _ok("\n".join(lines))


async def _h_pipeline_research(args: dict) -> dict:
    client = _require_auth()
    title = (args.get("title") or "").strip()
    sources = args.get("sources") or []
    if not title:
        return _err("title must be non-empty.")
    if not sources:
        return _err("sources must be a non-empty list.")
    source_type = args.get("source_type", "url")
    query = (args.get("query") or "").strip()
    studio = args.get("studio")
    confirm = args.get("confirm", False)

    trace: list[str] = [f"Pipeline: {title}"]
    # Stage 1: create notebook
    nb = client.create_notebook(title)
    nid = getattr(nb, "id", None) if nb else None
    if not nid:
        return _err("Failed to create notebook.")
    trace.append(f"  [1] notebook created: {nid}")

    # Stage 2: add sources
    added = 0
    failed = 0
    for src in sources:
        try:
            if source_type == "text":
                client.add_text_source(nid, src, title=f"{title} source {added+1}")
            else:
                client.add_url_source(nid, src)
            added += 1
        except Exception as e:
            log.warning(f"pipeline add source failed: {e}")
            failed += 1
    trace.append(f"  [2] sources: {added} added, {failed} failed")
    if added == 0:
        return _err("\n".join(trace + ["  No sources could be added. Aborting."]))

    # Stage 3: optional first query
    if query:
        try:
            result = client.query(nid, query)
            answer = (result or {}).get("answer") or "[no answer]"
            trace.append(f"  [3] query ok: {answer[:200]}")
        except Exception as e:
            trace.append(f"  [3] query failed: {e}")

    # Stage 4: optional studio artifact
    if studio and isinstance(studio, dict):
        if not confirm:
            trace.append("  [4] studio skipped (confirm=false)")
        else:
            studio_args = {
                "notebook_id": nid,
                "type": studio.get("type"),
                "confirm": True,
                "focus_prompt": studio.get("focus_prompt", ""),
            }
            studio_result = await _h_studio_create(studio_args)
            studio_text = ""
            try:
                studio_text = (studio_result.get("content") or [{}])[0].get("text", "")[:200]
            except Exception:
                pass
            marker = "FAIL" if studio_result.get("isError") else "OK"
            trace.append(f"  [4] studio {marker}: {studio_text}")

    if _cognitive and _cognitive.available:
        await _cognitive.capture_result(
            "pipeline_research",
            {"title": title, "notebook_id": nid, "sources_added": added},
        )
    return _ok("\n".join(trace))


# ── Handler Dispatch Table ───────────────────────────────────────────────

_HANDLERS = {
    # Notebooks
    "notebook_list": _h_notebook_list,
    "notebook_create": _h_notebook_create,
    "notebook_get": _h_notebook_get,
    "notebook_describe": _h_notebook_describe,
    "notebook_rename": _h_notebook_rename,
    "notebook_delete": _h_notebook_delete,
    "chat_configure": _h_chat_configure,
    # Sources
    "source_add": _h_source_add,
    "source_describe": _h_source_describe,
    "source_get_content": _h_source_get_content,
    "source_list_drive": _h_source_list_drive,
    "source_sync_drive": _h_source_sync_drive,
    "source_delete": _h_source_delete,
    "source_rename": _h_source_rename,
    # Query
    "notebook_query": _h_notebook_query,
    "notebook_query_start": _h_notebook_query_start,
    "notebook_query_status": _h_notebook_query_status,
    # Studio
    "studio_create": _h_studio_create,
    "studio_status": _h_studio_status,
    "studio_delete": _h_studio_delete,
    "artifact_rename": _h_artifact_rename,
    "slide_deck_revise": _h_slide_deck_revise,
    # Download / Export
    "download_artifact": _h_download_artifact,
    "export_artifact": _h_export_artifact,
    # Research
    "research_start": _h_research_start,
    "research_status": _h_research_status,
    "research_import": _h_research_import,
    # Sharing
    "notebook_share_status": _h_notebook_share_status,
    "notebook_share_public": _h_notebook_share_public,
    "notebook_share_invite": _h_notebook_share_invite,
    "notebook_share_batch": _h_notebook_share_batch,
    # Notes
    "note_create": _h_note_create,
    "note_list": _h_note_list,
    "note_update": _h_note_update,
    "note_delete": _h_note_delete,
    # Auth
    "save_auth_tokens": _h_save_auth_tokens,
    "refresh_auth": _h_refresh_auth,
    "auto_auth": _h_auto_auth,
    # Cognitive Intelligence
    "cognitive_enrich_query": _h_cognitive_enrich_query,
    "cognitive_search": _h_cognitive_search,
    "cognitive_insights": _h_cognitive_insights,
    "research_to_notebook": _h_research_to_notebook,
    "knowledge_evolution": _h_knowledge_evolution,
    "cognitive_auto_curate": _h_cognitive_auto_curate,
    # Phase 3: Convergence
    "cross_notebook_query": _h_cross_notebook_query,
    "batch_execute": _h_batch_execute,
    "pipeline_research": _h_pipeline_research,
}
