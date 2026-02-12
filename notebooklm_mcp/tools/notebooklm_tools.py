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


def set_api_client(client: NotebookLMAPIClient):
    global _api_client
    _api_client = client


def set_cognitive(layer: CognitiveLayer):
    global _cognitive
    _cognitive = layer


def _require_auth() -> NotebookLMAPIClient:
    if _api_client is None:
        raise AuthenticationError("Not authenticated. Call save_auth_tokens first with your NotebookLM cookies.")
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
                "title": {"type": "string", "description": "Notebook title (optional, defaults to 'Untitled notebook')"},
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
                "confirm": {"type": "boolean", "description": "Must be true to confirm deletion"},
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
                "goal": {"type": "string", "enum": ["default", "custom", "learning_guide"], "description": "Chat goal"},
                "custom_prompt": {"type": "string", "description": "Custom system prompt (required when goal='custom', max 10000 chars)"},
                "response_length": {"type": "string", "enum": ["default", "longer", "shorter"], "description": "Response length preference"},
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
                "source_type": {"type": "string", "enum": ["url", "text", "drive", "file"], "description": "Source type"},
                "url": {"type": "string", "description": "URL for 'url' type (web page or YouTube)"},
                "text": {"type": "string", "description": "Text content for 'text' type"},
                "title": {"type": "string", "description": "Title for 'text' type (default: 'Pasted Text')"},
                "document_id": {"type": "string", "description": "Google Doc/Sheets/Slides ID for 'drive' type"},
                "drive_title": {"type": "string", "description": "Document title for 'drive' type"},
                "mime_type": {"type": "string", "description": "MIME type for 'drive' type (default: application/vnd.google-apps.document)"},
                "file_path": {"type": "string", "description": "Local file path for 'file' type"},
                "wait": {"type": "boolean", "description": "Wait for source processing to complete (default: false)"},
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
                "confirm": {"type": "boolean", "description": "Must be true to confirm sync"},
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
                "confirm": {"type": "boolean", "description": "Must be true to confirm deletion"},
            },
            "required": ["source_id", "confirm"],
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
                    "type": "array", "items": {"type": "string"},
                    "description": "Optional list of source IDs to scope the query",
                },
                "conversation_id": {"type": "string", "description": "Optional conversation ID for follow-up questions"},
            },
            "required": ["notebook_id", "query"],
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
                    "enum": ["audio", "video", "report", "flashcards", "quiz", "infographic", "slide_deck", "data_table"],
                    "description": "Content type to generate",
                },
                "source_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional source IDs (default: all)"},
                "focus_prompt": {"type": "string", "description": "Optional focus/customization prompt"},
                "language": {"type": "string", "description": "Language code (default: en)"},
                "audio_format": {"type": "string", "enum": ["deep_dive", "brief", "critique", "debate"], "description": "Audio format"},
                "audio_length": {"type": "string", "enum": ["short", "default", "long"], "description": "Audio length"},
                "video_format": {"type": "string", "enum": ["explainer", "brief"], "description": "Video format"},
                "video_style": {
                    "type": "string",
                    "enum": ["auto_select", "custom", "classic", "whiteboard", "kawaii", "anime", "watercolor", "retro_print", "heritage", "paper_craft"],
                    "description": "Video visual style",
                },
                "report_format": {
                    "type": "string",
                    "enum": ["Briefing Doc", "Study Guide", "Blog Post", "Create Your Own"],
                    "description": "Report format",
                },
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"], "description": "Flashcard/quiz difficulty"},
                "question_count": {"type": "integer", "description": "Number of quiz questions (default: 2)"},
                "orientation": {"type": "string", "enum": ["landscape", "portrait", "square"], "description": "Infographic orientation"},
                "detail_level": {"type": "string", "enum": ["concise", "standard", "detailed"], "description": "Infographic detail level"},
                "slide_format": {"type": "string", "enum": ["detailed_deck", "presenter_slides"], "description": "Slide deck format"},
                "slide_length": {"type": "string", "enum": ["short", "default"], "description": "Slide deck length"},
                "confirm": {"type": "boolean", "description": "Must be true to start generation"},
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
                "notebook_id": {"type": "string", "description": "Notebook UUID (optional, for mind map fallback)"},
                "confirm": {"type": "boolean", "description": "Must be true to confirm deletion"},
            },
            "required": ["artifact_id", "confirm"],
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
                "title": {"type": "string", "description": "Export title (default: 'NotebookLM Export')"},
                "export_type": {"type": "string", "enum": ["docs", "sheets"], "description": "Export to Google Docs or Sheets"},
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
                "source": {"type": "string", "enum": ["web", "drive"], "description": "Search source (default: web)"},
                "mode": {"type": "string", "enum": ["fast", "deep"], "description": "Research mode (default: fast)"},
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
                "task_id": {"type": "string", "description": "Optional task ID from research_start"},
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
                    "type": "array", "items": {"type": "integer"},
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
                "is_public": {"type": "boolean", "description": "True to enable public access, false to disable"},
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
                "role": {"type": "string", "enum": ["editor", "viewer"], "description": "Role (default: viewer)"},
                "notify": {"type": "boolean", "description": "Send email notification (default: true)"},
                "message": {"type": "string", "description": "Optional message in the invite email"},
            },
            "required": ["notebook_id", "email"],
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
                "title": {"type": "string", "description": "Note title (default: 'New Note')"},
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
                "confirm": {"type": "boolean", "description": "Must be true to confirm deletion"},
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
                "cookies": {"type": "string", "description": "Full cookie header string from Chrome DevTools"},
            },
            "required": ["cookies"],
        },
    },
    {
        "name": "refresh_auth",
        "description": "Refresh CSRF token and session ID by fetching the NotebookLM homepage.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
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
                "source_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional source IDs to scope"},
                "max_context_items": {"type": "integer", "description": "Max enrichment items (default: 5)"},
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
                "limit": {"type": "integer", "description": "Max results (default: 10)"},
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
                "limit": {"type": "integer", "description": "Max insights (default: 10)"},
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
                "session_id": {"type": "string", "description": "ResearchGravity session ID"},
                "notebook_id": {"type": "string", "description": "Optional existing notebook UUID (creates new if omitted)"},
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
                "arc_id": {"type": "string", "description": "Optional arc ID to auto-curate (from suggestions)"},
            },
            "required": [],
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
        lines.append(f"    ID: {nb.id} | Sources: {nb.source_count} | Modified: {nb.modified_at or 'unknown'}")
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
        status = "ready" if s.get("status") == 2 else "processing" if s.get("status") == 1 else "error" if s.get("status") == 3 else "unknown"
        lines.append(f"  {s['title']} [{s['source_type_name']}] ({status})")
        lines.append(f"    ID: {s['id']}" + (f" | URL: {s['url']}" if s.get("url") else ""))
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
    return _ok(f"Renamed to: {args['new_title']}") if ok else _err("Failed to rename notebook.")


async def _h_notebook_delete(args: dict) -> dict:
    if not args.get("confirm"):
        return _err("Deletion requires confirm=true. This is IRREVERSIBLE.")
    client = _require_auth()
    ok = client.delete_notebook(args["notebook_id"])
    return _ok(f"Deleted notebook {args['notebook_id']}") if ok else _err("Failed to delete notebook.")


async def _h_chat_configure(args: dict) -> dict:
    client = _require_auth()
    result = client.configure_chat(
        args["notebook_id"],
        goal=args.get("goal", "default"),
        custom_prompt=args.get("custom_prompt"),
        response_length=args.get("response_length", "default"),
    )
    if result.get("status") == "success":
        return _ok(f"Chat configured: goal={result['goal']}, length={result['response_length']}")
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
        result = client.add_text_source(nid, text, title=args.get("title", "Pasted Text"), wait=wait)
    elif stype == "drive":
        doc_id = args.get("document_id")
        if not doc_id:
            return _err("'document_id' is required for source_type='drive'")
        result = client.add_drive_source(
            nid, doc_id, title=args.get("drive_title", "Drive Document"),
            mime_type=args.get("mime_type", "application/vnd.google-apps.document"), wait=wait,
        )
    elif stype == "file":
        fp = args.get("file_path")
        if not fp:
            return _err("'file_path' is required for source_type='file'")
        result = client.add_file(nid, fp, wait=wait)
    else:
        return _err(f"Invalid source_type: {stype}")

    if result:
        return _ok(f"Source added: {result.get('title', 'Unknown')}\n  ID: {result.get('id')}")
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
        lines.append(f"\n... (truncated, {result['char_count'] - 5000} chars remaining)")
    return _ok("\n".join(lines))


async def _h_source_list_drive(args: dict) -> dict:
    client = _require_auth()
    sources = client.get_notebook_sources_with_types(args["notebook_id"])
    lines = [f"Sources ({len(sources)}):\n"]
    for s in sources:
        status = "ready" if s.get("status") == 2 else "processing" if s.get("status") == 1 else "error" if s.get("status") == 3 else "unknown"
        sync_info = ""
        if s.get("can_sync"):
            fresh = client.check_source_freshness(s["id"])
            sync_info = " [STALE - needs sync]" if fresh is False else " [fresh]" if fresh else ""
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
    return _ok(f"Deleted source {args['source_id']}") if ok else _err("Failed to delete source.")


# ── Query Handler ────────────────────────────────────────────────────────

async def _h_notebook_query(args: dict) -> dict:
    client = _require_auth()
    result = client.query(
        args["notebook_id"], args["query"],
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
        await _cognitive.capture_result("notebook_query", {
            "query": args["query"], "answer": result["answer"][:500],
            "notebook_id": args["notebook_id"],
        })
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
        result = client.create_audio_overview(nid, sids, format_code=fc, length_code=lc, language=lang, focus_prompt=focus)
    elif stype == "video":
        fc = constants.VIDEO_FORMATS.get_code(args.get("video_format", "explainer"))
        vs = constants.VIDEO_STYLES.get_code(args.get("video_style", "auto_select"))
        result = client.create_video_overview(nid, sids, format_code=fc, visual_style_code=vs, language=lang, focus_prompt=focus)
    elif stype == "report":
        rf = args.get("report_format", "Briefing Doc")
        result = client.create_report(nid, sids, report_format=rf, custom_prompt=focus, language=lang)
    elif stype == "flashcards":
        dc = constants.FLASHCARD_DIFFICULTIES.get_code(args.get("difficulty", "medium"))
        result = client.create_flashcards(nid, sids, difficulty_code=dc)
    elif stype == "quiz":
        dc = constants.FLASHCARD_DIFFICULTIES.get_code(args.get("difficulty", "medium"))
        qc = args.get("question_count", 2)
        result = client.create_quiz(nid, sids, question_count=qc, difficulty=dc)
    elif stype == "infographic":
        oc = constants.INFOGRAPHIC_ORIENTATIONS.get_code(args.get("orientation", "landscape"))
        dl = constants.INFOGRAPHIC_DETAILS.get_code(args.get("detail_level", "standard"))
        result = client.create_infographic(nid, sids, orientation_code=oc, detail_level_code=dl, language=lang, focus_prompt=focus)
    elif stype == "slide_deck":
        sf = constants.SLIDE_DECK_FORMATS.get_code(args.get("slide_format", "detailed_deck"))
        sl = constants.SLIDE_DECK_LENGTHS.get_code(args.get("slide_length", "default"))
        result = client.create_slide_deck(nid, sids, format_code=sf, length_code=sl, language=lang, focus_prompt=focus)
    elif stype == "data_table":
        result = client.create_data_table(nid, sids, description=focus, language=lang)
    else:
        return _err(f"Unknown studio type: {stype}")

    if result:
        return _ok(f"Studio '{stype}' creation started.\n  Artifact ID: {result['artifact_id']}\n  Status: {result['status']}")
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
    ok = client.delete_studio_artifact(args["artifact_id"], notebook_id=args.get("notebook_id"))
    return _ok(f"Deleted artifact {args['artifact_id']}") if ok else _err("Failed to delete artifact.")


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
        args["notebook_id"], args["artifact_id"],
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
        args["notebook_id"], args["query"],
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
    result = client.poll_research(args["notebook_id"], target_task_id=args.get("task_id"))
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
            lines.append(f"  [{s['index']}] {s['title']}" + (f" - {s['url']}" if s.get("url") else ""))
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
        args["notebook_id"], args["email"],
        role=args.get("role", "viewer"),
        notify=args.get("notify", True),
        message=args.get("message", ""),
    )
    return _ok(f"Invited {args['email']} as {args.get('role', 'viewer')}") if ok else _err("Failed to invite collaborator.")


# ── Note Handlers ────────────────────────────────────────────────────────

async def _h_note_create(args: dict) -> dict:
    client = _require_auth()
    result = client.create_note(args["notebook_id"], args["content"], title=args.get("title"))
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
        args["note_id"], content=args.get("content"), title=args.get("title"),
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
    return _ok(f"Deleted note {args['note_id']}") if ok else _err("Failed to delete note.")


# ── Auth Handlers ────────────────────────────────────────────────────────

async def _h_save_auth_tokens(args: dict) -> dict:
    import os
    global _api_client
    cookies = args["cookies"]
    # Also save to env for persistence within session
    os.environ["NOTEBOOKLM_COOKIES"] = cookies
    try:
        _api_client = NotebookLMAPIClient(cookies=cookies)
        # Verify by listing notebooks
        nbs = _api_client.list_notebooks()
        return _ok(f"Authenticated. Found {len(nbs)} notebooks.")
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


# ── Cognitive Handlers ───────────────────────────────────────────────────

async def _h_cognitive_enrich_query(args: dict) -> dict:
    _require_auth()
    if not _cognitive or not _cognitive.available:
        return _err("Cognitive layer not available. PostgreSQL + coherence_engine required.")
    result = await _cognitive.enriched_query(
        args["notebook_id"], args["query"],
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
    results = await _cognitive.cognitive_search(args["query"], limit=args.get("limit", 10))
    if not results:
        return _ok("No results found.")
    lines = [f"Found {len(results)} results:\n"]
    for r in results:
        score = r.get("score", r.get("relevance_score", 0))
        platform = r.get("platform", "unknown")
        content = r.get("content", "")[:200]
        lines.append(f"  [{score:.3f}] [{platform}] {content}")
    return _ok("\n".join(lines))


async def _h_cognitive_insights(args: dict) -> dict:
    if not _cognitive or not _cognitive.available:
        return _err("Cognitive layer not available.")
    insights = await _cognitive.get_due_insights(limit=args.get("limit", 10))
    if not insights:
        return _ok("No insights due for review.")
    lines = [f"Due for review ({len(insights)}):\n"]
    for i in insights:
        lines.append(f"  - {i.get('content', '')[:200]}")
    return _ok("\n".join(lines))


async def _h_research_to_notebook(args: dict) -> dict:
    _require_auth()
    if not _cognitive or not _cognitive.available:
        return _err("Cognitive layer not available.")
    result = await _cognitive.import_research_session(
        args["session_id"], notebook_id=args.get("notebook_id"),
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
        args["notebook_id"], args["query"], args["response"],
    )
    if result.get("tracked"):
        lines = [
            f"Knowledge tracked.",
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
            lines.append(f"    Arc ID: {s['arc_id']} | Significance: {s['significance']:.2f} | Platforms: {s['platforms']} | Moments: {s['moments']}")
        lines.append("\nCall cognitive_auto_curate with arc_id to create a notebook.")
        return _ok("\n".join(lines))


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
    # Query
    "notebook_query": _h_notebook_query,
    # Studio
    "studio_create": _h_studio_create,
    "studio_status": _h_studio_status,
    "studio_delete": _h_studio_delete,
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
    # Notes
    "note_create": _h_note_create,
    "note_list": _h_note_list,
    "note_update": _h_note_update,
    "note_delete": _h_note_delete,
    # Auth
    "save_auth_tokens": _h_save_auth_tokens,
    "refresh_auth": _h_refresh_auth,
    # Cognitive Intelligence
    "cognitive_enrich_query": _h_cognitive_enrich_query,
    "cognitive_search": _h_cognitive_search,
    "cognitive_insights": _h_cognitive_insights,
    "research_to_notebook": _h_research_to_notebook,
    "knowledge_evolution": _h_knowledge_evolution,
    "cognitive_auto_curate": _h_cognitive_auto_curate,
}
