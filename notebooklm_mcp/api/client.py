"""NotebookLM API client — full HTTP/RPC access via batchexecute protocol.

Flattened from jacob-bd/notebooklm-mcp-cli mixin architecture into a single class.
All domain methods (notebooks, sources, conversation, studio, research, sharing,
notes, exports) are methods on NotebookLMAPIClient.
"""

import json
import logging
import os
import random
import re
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from . import constants
from .constants import RPC_NAMES

logger = logging.getLogger("notebooklm_mcp.api")
logger.setLevel(logging.WARNING)

# Timeouts
DEFAULT_TIMEOUT = 30.0
SOURCE_ADD_TIMEOUT = 120.0
QUERY_TIMEOUT = 120.0

# Retry config
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 16.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    query: str
    answer: str
    turn_number: int


@dataclass
class Notebook:
    id: str
    title: str
    source_count: int
    sources: list[dict]
    is_owned: bool = True
    is_shared: bool = False
    created_at: str | None = None
    modified_at: str | None = None

    @property
    def url(self) -> str:
        return f"https://notebooklm.google.com/notebook/{self.id}"


@dataclass
class Collaborator:
    email: str
    role: str
    is_pending: bool = False
    display_name: str | None = None


@dataclass
class ShareStatus:
    is_public: bool
    access_level: str
    collaborators: list[Collaborator]
    public_link: str | None = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class NotebookLMError(Exception):
    pass


class AuthenticationError(NotebookLMError):
    pass


class ArtifactError(NotebookLMError):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_timestamp(ts_array: list | None) -> str | None:
    if not ts_array or not isinstance(ts_array, list) or len(ts_array) < 1:
        return None
    try:
        seconds = ts_array[0]
        if not isinstance(seconds, (int, float)):
            return None
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, OSError, OverflowError):
        return None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


def _extract_cookies_from_string(cookie_str: str) -> dict[str, str]:
    cookies = {}
    for item in cookie_str.split(";"):
        if "=" in item:
            name, value = item.strip().split("=", 1)
            cookies[name] = value
    return cookies


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class NotebookLMAPIClient:
    """Full HTTP/RPC client for NotebookLM internal API.

    Provides 30+ methods covering notebooks, sources, conversations, studio
    content, research, sharing, notes, and exports.

    Auth: pass cookies as dict[str,str], list[dict], or cookie header string.
    CSRF token and session ID are auto-extracted from homepage.
    """

    BASE_URL = "https://notebooklm.google.com"
    BATCHEXECUTE_URL = f"{BASE_URL}/_/LabsTailwindUi/data/batchexecute"
    UPLOAD_URL = f"{BASE_URL}/upload/_/"

    # RPC IDs -----------------------------------------------------------
    RPC_LIST_NOTEBOOKS = "wXbhsf"
    RPC_GET_NOTEBOOK = "rLM1Ne"
    RPC_CREATE_NOTEBOOK = "CCqFvf"
    RPC_RENAME_NOTEBOOK = "s0tc2d"
    RPC_DELETE_NOTEBOOK = "WWINqb"
    RPC_ADD_SOURCE = "izAoDd"
    RPC_ADD_SOURCE_FILE = "o4cbdc"
    RPC_GET_SOURCE = "hizoJc"
    RPC_CHECK_FRESHNESS = "yR9Yof"
    RPC_SYNC_DRIVE = "FLmJqe"
    RPC_DELETE_SOURCE = "tGMBJ"
    RPC_GET_SUMMARY = "VfAZjd"
    RPC_GET_SOURCE_GUIDE = "tr032e"
    RPC_START_FAST_RESEARCH = "Ljjv0c"
    RPC_START_DEEP_RESEARCH = "QA9ei"
    RPC_POLL_RESEARCH = "e3bVqc"
    RPC_IMPORT_RESEARCH = "LBwxtb"
    RPC_CREATE_STUDIO = "R7cb6c"
    RPC_POLL_STUDIO = "gArtLc"
    RPC_DELETE_STUDIO = "V5N4be"
    RPC_RENAME_ARTIFACT = "rc3d8d"
    RPC_GET_INTERACTIVE_HTML = "v9rmvd"
    RPC_GENERATE_MIND_MAP = "yyryJe"
    RPC_SAVE_MIND_MAP = "CYK0Xb"
    RPC_LIST_MIND_MAPS = "cFji9"
    RPC_DELETE_MIND_MAP = "AH0mwd"
    RPC_CREATE_NOTE = "CYK0Xb"
    RPC_GET_NOTES = "cFji9"
    RPC_UPDATE_NOTE = "cYAfTb"
    RPC_DELETE_NOTE = "AH0mwd"
    RPC_SHARE_NOTEBOOK = "QDyure"
    RPC_GET_SHARE_STATUS = "JFMDGd"
    RPC_EXPORT_ARTIFACT = "Krh3pd"

    QUERY_ENDPOINT = (
        "/_/LabsTailwindUi/data/google.internal.labs.tailwind.orchestration"
        ".v1.LabsTailwindOrchestrationService/GenerateFreeFormStreamed"
    )

    _PAGE_FETCH_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }

    # Source processing status codes
    SOURCE_STATUS_PROCESSING = 1
    SOURCE_STATUS_READY = 2
    SOURCE_STATUS_ERROR = 3
    SOURCE_STATUS_PREPARING = 5

    # =====================================================================
    # Lifecycle
    # =====================================================================

    def __init__(
        self,
        cookies: dict[str, str] | list[dict] | str = "",
        csrf_token: str = "",
        session_id: str = "",
    ):
        # Normalize cookies
        if isinstance(cookies, str):
            if cookies:
                self.cookies: dict[str, str] | list[dict] = _extract_cookies_from_string(cookies)
            else:
                self.cookies = {}
        else:
            self.cookies = cookies

        self.csrf_token = csrf_token
        self._session_id = session_id
        self._client: httpx.Client | None = None
        self._conversation_cache: dict[str, list[ConversationTurn]] = {}
        self._reqid_counter = random.randint(100000, 999999)

        if not self.csrf_token and self.cookies:
            self._refresh_auth_tokens()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._client:
            self._client.close()
            self._client = None

    # =====================================================================
    # Cookie / HTTP client
    # =====================================================================

    def _get_httpx_cookies(self) -> httpx.Cookies:
        jar = httpx.Cookies()
        if isinstance(self.cookies, list):
            for c in self.cookies:
                name, value = c.get("name"), c.get("value")
                domain = c.get("domain", ".google.com")
                path = c.get("path", "/")
                if name and value:
                    jar.set(name, value, domain=domain, path=path)
                    if domain == ".google.com":
                        jar.set(name, value, domain=".googleusercontent.com", path=path)
        else:
            for name, value in self.cookies.items():
                jar.set(name, value, domain=".google.com")
                jar.set(name, value, domain=".googleusercontent.com")
        return jar

    def _get_cookie_header(self) -> str:
        if isinstance(self.cookies, list):
            pairs = {c["name"]: c["value"] for c in self.cookies if "name" in c and "value" in c}
            return "; ".join(f"{k}={v}" for k, v in pairs.items())
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items())

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                cookies=self._get_httpx_cookies(),
                headers={
                    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                    "Origin": self.BASE_URL,
                    "Referer": f"{self.BASE_URL}/",
                    "X-Same-Domain": "1",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                },
                timeout=DEFAULT_TIMEOUT,
            )
            if self.csrf_token:
                self._client.headers["X-Goog-Csrf-Token"] = self.csrf_token
        return self._client

    # =====================================================================
    # RPC protocol
    # =====================================================================

    def _build_request_body(self, rpc_id: str, params: Any) -> str:
        params_json = json.dumps(params, separators=(",", ":"))
        f_req = [[[rpc_id, params_json, None, "generic"]]]
        f_req_json = json.dumps(f_req, separators=(",", ":"))
        parts = [f"f.req={urllib.parse.quote(f_req_json, safe='')}"]
        if self.csrf_token:
            parts.append(f"at={urllib.parse.quote(self.csrf_token, safe='')}")
        return "&".join(parts) + "&"

    def _build_url(self, rpc_id: str, source_path: str = "/") -> str:
        params = {
            "rpcids": rpc_id,
            "source-path": source_path,
            "bl": os.environ.get("NOTEBOOKLM_BL", "boq_labs-tailwind-frontend_20260108.06_p0"),
            "hl": "en",
            "rt": "c",
        }
        if self._session_id:
            params["f.sid"] = self._session_id
        return f"{self.BATCHEXECUTE_URL}?{urllib.parse.urlencode(params)}"

    def _parse_response(self, text: str) -> list:
        if text.startswith(")]}'"):
            text = text[4:]
        lines = text.strip().split("\n")
        results = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            try:
                int(line)  # byte count
                i += 1
                if i < len(lines):
                    try:
                        results.append(json.loads(lines[i]))
                    except json.JSONDecodeError:
                        pass
                i += 1
            except ValueError:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
                i += 1
        return results

    def _extract_rpc_result(self, parsed: list, rpc_id: str) -> Any:
        for chunk in parsed:
            if isinstance(chunk, list):
                for item in chunk:
                    if isinstance(item, list) and len(item) >= 3:
                        if item[0] == "wrb.fr" and item[1] == rpc_id:
                            if (
                                len(item) > 6
                                and item[6] == "generic"
                                and isinstance(item[5], list)
                                and 16 in item[5]
                            ):
                                raise AuthenticationError("RPC Error 16: Authentication expired")
                            result_str = item[2]
                            if isinstance(result_str, str):
                                try:
                                    return json.loads(result_str)
                                except json.JSONDecodeError:
                                    return result_str
                            return result_str
        return None

    def _call_rpc(
        self,
        rpc_id: str,
        params: Any,
        path: str = "/",
        timeout: float | None = None,
        _retry: bool = False,
        _server_retry: int = 0,
    ) -> Any:
        """Execute an RPC call with 2-layer auth recovery and server-error retry."""
        client = self._get_client()
        body = self._build_request_body(rpc_id, params)
        url = self._build_url(rpc_id, path)

        try:
            kw: dict[str, Any] = {}
            if timeout:
                kw["timeout"] = timeout
            response = client.post(url, content=body, **kw)
            response.raise_for_status()
            parsed = self._parse_response(response.text)
            return self._extract_rpc_result(parsed, rpc_id)

        except httpx.HTTPStatusError as e:
            # Retry transient server errors
            if _is_retryable(e) and _server_retry < MAX_RETRIES:
                delay = min(BASE_DELAY * (2 ** _server_retry), MAX_DELAY)
                logger.warning(
                    f"Server error {e.response.status_code}, retry {_server_retry + 1}/{MAX_RETRIES} in {delay:.1f}s"
                )
                time.sleep(delay)
                return self._call_rpc(rpc_id, params, path, timeout, _retry, _server_retry + 1)

            if e.response.status_code not in (401, 403):
                raise
            # Fall through to auth recovery

        except AuthenticationError:
            pass  # Fall through to auth recovery

        # Auth recovery: refresh CSRF token
        if not _retry:
            try:
                self._refresh_auth_tokens()
                self._client = None
                return self._call_rpc(rpc_id, params, path, timeout, _retry=True)
            except ValueError:
                pass

        raise AuthenticationError(
            "Authentication expired. Re-extract cookies from Chrome DevTools."
        )

    # =====================================================================
    # Auth management
    # =====================================================================

    def _refresh_auth_tokens(self) -> None:
        """Fetch NotebookLM homepage to extract CSRF token (SNlM0e) and session ID (FdrFJe)."""
        cookies = self._get_httpx_cookies()
        with httpx.Client(
            cookies=cookies, headers=self._PAGE_FETCH_HEADERS, follow_redirects=True, timeout=15.0
        ) as client:
            response = client.get(f"{self.BASE_URL}/")

            if "accounts.google.com" in str(response.url):
                raise ValueError("Authentication expired — redirected to Google login.")

            if response.status_code != 200:
                raise ValueError(f"Failed to fetch NotebookLM page: HTTP {response.status_code}")

            html = response.text
            csrf_match = re.search(r'"SNlM0e":"([^"]+)"', html)
            if not csrf_match:
                raise ValueError("Could not extract CSRF token from page.")

            self.csrf_token = csrf_match.group(1)

            sid_match = re.search(r'"FdrFJe":"([^"]+)"', html)
            if sid_match:
                self._session_id = sid_match.group(1)

    def update_cookies(self, cookies: dict[str, str] | list[dict] | str) -> None:
        """Update cookies and force re-init."""
        if isinstance(cookies, str):
            self.cookies = _extract_cookies_from_string(cookies) if cookies else {}
        else:
            self.cookies = cookies
        self.csrf_token = ""
        self._session_id = ""
        self._client = None
        if self.cookies:
            self._refresh_auth_tokens()

    # =====================================================================
    # Notebook operations
    # =====================================================================

    def list_notebooks(self) -> list[Notebook]:
        params = [None, 1, None, [2]]
        client = self._get_client()
        body = self._build_request_body(self.RPC_LIST_NOTEBOOKS, params)
        url = self._build_url(self.RPC_LIST_NOTEBOOKS)
        response = client.post(url, content=body)
        response.raise_for_status()
        parsed = self._parse_response(response.text)
        result = self._extract_rpc_result(parsed, self.RPC_LIST_NOTEBOOKS)

        notebooks = []
        if result and isinstance(result, list):
            nb_list = result[0] if result and isinstance(result[0], list) else result
            for nb in nb_list:
                if not isinstance(nb, list) or len(nb) < 3:
                    continue
                title = nb[0] if isinstance(nb[0], str) else "Untitled"
                sources_data = nb[1] if len(nb) > 1 and isinstance(nb[1], list) else []
                notebook_id = nb[2] if len(nb) > 2 else None
                if not notebook_id:
                    continue

                is_owned, is_shared = True, False
                created_at = modified_at = None
                if len(nb) > 5 and isinstance(nb[5], list) and len(nb[5]) > 0:
                    meta = nb[5]
                    is_owned = meta[0] == constants.OWNERSHIP_MINE
                    if len(meta) > 1:
                        is_shared = bool(meta[1])
                    if len(meta) > 5:
                        modified_at = _parse_timestamp(meta[5])
                    if len(meta) > 8:
                        created_at = _parse_timestamp(meta[8])

                sources = []
                for src in sources_data:
                    if isinstance(src, list) and len(src) >= 2:
                        sid = src[0][0] if isinstance(src[0], list) and src[0] else src[0]
                        sources.append({"id": sid, "title": src[1] if len(src) > 1 else "Untitled"})

                notebooks.append(Notebook(
                    id=notebook_id, title=title, source_count=len(sources),
                    sources=sources, is_owned=is_owned, is_shared=is_shared,
                    created_at=created_at, modified_at=modified_at,
                ))
        return notebooks

    def get_notebook(self, notebook_id: str) -> dict | None:
        return self._call_rpc(
            self.RPC_GET_NOTEBOOK, [notebook_id, None, [2], None, 0],
            f"/notebook/{notebook_id}",
        )

    def get_notebook_summary(self, notebook_id: str) -> dict[str, Any]:
        result = self._call_rpc(self.RPC_GET_SUMMARY, [notebook_id, [2]], f"/notebook/{notebook_id}")
        summary = ""
        topics = []
        if result and isinstance(result, list):
            if len(result) > 0 and isinstance(result[0], list) and len(result[0]) > 0:
                summary = result[0][0]
            if len(result) > 1 and result[1]:
                td = result[1][0] if isinstance(result[1], list) and len(result[1]) > 0 else []
                for t in td:
                    if isinstance(t, list) and len(t) >= 2:
                        topics.append({"question": t[0], "prompt": t[1]})
        return {"summary": summary, "suggested_topics": topics}

    def create_notebook(self, title: str = "") -> Notebook | None:
        params = [title, None, None, [2], [1, None, None, None, None, None, None, None, None, None, [1]]]
        result = self._call_rpc(self.RPC_CREATE_NOTEBOOK, params)
        if result and isinstance(result, list) and len(result) >= 3:
            nid = result[2]
            if nid:
                return Notebook(id=nid, title=title or "Untitled notebook", source_count=0, sources=[])
        return None

    def rename_notebook(self, notebook_id: str, new_title: str) -> bool:
        params = [notebook_id, [[None, None, None, [None, new_title]]]]
        result = self._call_rpc(self.RPC_RENAME_NOTEBOOK, params, f"/notebook/{notebook_id}")
        return result is not None

    def configure_chat(
        self,
        notebook_id: str,
        goal: str = "default",
        custom_prompt: str | None = None,
        response_length: str = "default",
    ) -> dict[str, Any]:
        goal_code = constants.CHAT_GOALS.get_code(goal)
        if goal == "custom":
            if not custom_prompt:
                raise ValueError("custom_prompt required when goal='custom'")
            if len(custom_prompt) > 10000:
                raise ValueError(f"custom_prompt exceeds 10000 chars ({len(custom_prompt)})")
        length_code = constants.CHAT_RESPONSE_LENGTHS.get_code(response_length)
        goal_setting = [goal_code, custom_prompt] if goal == "custom" and custom_prompt else [goal_code]
        chat_settings = [goal_setting, [length_code]]
        params = [notebook_id, [[None, None, None, None, None, None, None, chat_settings]]]
        result = self._call_rpc(self.RPC_RENAME_NOTEBOOK, params, f"/notebook/{notebook_id}")
        if result:
            settings = result[7] if len(result) > 7 else None
            return {"status": "success", "notebook_id": notebook_id, "goal": goal,
                    "custom_prompt": custom_prompt if goal == "custom" else None,
                    "response_length": response_length, "raw_settings": settings}
        return {"status": "error", "error": "Failed to configure chat settings"}

    def delete_notebook(self, notebook_id: str) -> bool:
        client = self._get_client()
        body = self._build_request_body(self.RPC_DELETE_NOTEBOOK, [[notebook_id], [2]])
        url = self._build_url(self.RPC_DELETE_NOTEBOOK)
        response = client.post(url, content=body)
        response.raise_for_status()
        parsed = self._parse_response(response.text)
        return self._extract_rpc_result(parsed, self.RPC_DELETE_NOTEBOOK) is not None

    # =====================================================================
    # Source operations
    # =====================================================================

    def get_notebook_sources_with_types(self, notebook_id: str) -> list[dict]:
        result = self.get_notebook(notebook_id)
        sources = []
        if result and isinstance(result, list) and len(result) >= 1:
            nb_data = result[0] if isinstance(result[0], list) else result
            src_data = nb_data[1] if len(nb_data) > 1 and isinstance(nb_data[1], list) else []
            for src in src_data:
                if not isinstance(src, list) or len(src) < 3:
                    continue
                sid = src[0][0] if src[0] and isinstance(src[0], list) else None
                title = src[1] if len(src) > 1 else "Untitled"
                meta = src[2] if len(src) > 2 and isinstance(src[2], list) else []
                stype = meta[4] if len(meta) > 4 else None
                drive_id = meta[0][0] if len(meta) > 0 and isinstance(meta[0], list) and meta[0] else None
                can_sync = drive_id is not None and stype in (
                    constants.SOURCE_TYPE_GOOGLE_DOCS, constants.SOURCE_TYPE_GOOGLE_OTHER)
                url = None
                if len(meta) > 7 and isinstance(meta[7], list) and len(meta[7]) > 0:
                    url = meta[7][0]
                status = self.SOURCE_STATUS_READY
                if len(src) > 3 and isinstance(src[3], list) and len(src[3]) > 1:
                    status = src[3][1] if isinstance(src[3][1], int) else status
                sources.append({
                    "id": sid, "title": title, "source_type": stype,
                    "source_type_name": constants.SOURCE_TYPES.get_name(stype),
                    "url": url, "drive_doc_id": drive_id, "can_sync": can_sync, "status": status,
                })
        return sources

    def add_url_source(self, notebook_id: str, url: str, wait: bool = False, wait_timeout: float = 120.0) -> dict | None:
        is_yt = "youtube.com" in url.lower() or "youtu.be" in url.lower()
        if is_yt:
            sd = [None, None, None, None, None, None, None, [url], None, None, 1]
        else:
            sd = [None, None, [url], None, None, None, None, None, None, None, 1]
        params = [[sd], notebook_id, [2], [1, None, None, None, None, None, None, None, None, None, [1]]]
        result = self._call_rpc(self.RPC_ADD_SOURCE, params, f"/notebook/{notebook_id}", timeout=SOURCE_ADD_TIMEOUT)
        sr = self._parse_source_add_result(result)
        if sr and wait:
            return self.wait_for_source_ready(notebook_id, sr["id"], wait_timeout)
        return sr

    def add_text_source(self, notebook_id: str, text: str, title: str = "Pasted Text",
                        wait: bool = False, wait_timeout: float = 120.0) -> dict | None:
        sd = [None, [title, text], None, 2, None, None, None, None, None, None, 1]
        params = [[sd], notebook_id, [2], [1, None, None, None, None, None, None, None, None, None, [1]]]
        result = self._call_rpc(self.RPC_ADD_SOURCE, params, f"/notebook/{notebook_id}", timeout=SOURCE_ADD_TIMEOUT)
        sr = self._parse_source_add_result(result)
        if sr and wait:
            return self.wait_for_source_ready(notebook_id, sr["id"], wait_timeout)
        return sr

    def add_drive_source(self, notebook_id: str, document_id: str, title: str,
                         mime_type: str = "application/vnd.google-apps.document",
                         wait: bool = False, wait_timeout: float = 120.0) -> dict | None:
        sd = [[document_id, mime_type, 1, title], None, None, None, None, None, None, None, None, None, 1]
        params = [[sd], notebook_id, [2], [1, None, None, None, None, None, None, None, None, None, [1]]]
        result = self._call_rpc(self.RPC_ADD_SOURCE, params, f"/notebook/{notebook_id}", timeout=SOURCE_ADD_TIMEOUT)
        sr = self._parse_source_add_result(result)
        if sr and wait:
            return self.wait_for_source_ready(notebook_id, sr["id"], wait_timeout)
        return sr

    def add_file(self, notebook_id: str, file_path: str | Path, wait: bool = False,
                 wait_timeout: float = 120.0) -> dict:
        fp = Path(file_path)
        if not fp.exists() or not fp.is_file():
            raise ValueError(f"File not found or not a regular file: {fp}")
        if fp.stat().st_size == 0:
            raise ValueError(f"File is empty: {fp}")
        supported = {'.pdf', '.txt', '.md', '.docx', '.csv', '.mp3', '.mp4', '.jpg', '.jpeg', '.png'}
        if fp.suffix.lower() not in supported:
            raise ValueError(f"Unsupported file type: {fp.suffix}. Supported: {', '.join(sorted(supported))}")

        # Step 1: Register
        params = [[[fp.name]], notebook_id, [2], [1, None, None, None, None, None, None, None, None, None, [1]]]
        result = self._call_rpc(self.RPC_ADD_SOURCE_FILE, params, f"/notebook/{notebook_id}", timeout=60.0)
        source_id = self._extract_nested_id(result)
        if not source_id:
            raise RuntimeError("Failed to register file source")

        # Step 2: Start resumable upload
        upload_url = self._start_resumable_upload(notebook_id, fp.name, fp.stat().st_size, source_id)

        # Step 3: Upload content
        self._upload_file_streaming(upload_url, fp)

        sr = {"id": source_id, "title": fp.name}
        if wait:
            return self.wait_for_source_ready(notebook_id, source_id, wait_timeout)
        return sr

    def check_source_freshness(self, source_id: str) -> bool | None:
        result = self._call_rpc(self.RPC_CHECK_FRESHNESS, [None, [source_id], [2]])
        if result and isinstance(result, list) and len(result) > 0:
            inner = result[0] if result else []
            if isinstance(inner, list) and len(inner) >= 2:
                return inner[1]
        return None

    def sync_drive_source(self, source_id: str) -> dict | None:
        result = self._call_rpc(self.RPC_SYNC_DRIVE, [None, [source_id], [2]])
        if result and isinstance(result, list) and len(result) > 0:
            sd = result[0] if result else []
            if isinstance(sd, list) and len(sd) >= 3:
                sid = sd[0][0] if sd[0] else None
                title = sd[1] if len(sd) > 1 else "Unknown"
                return {"id": sid, "title": title}
        return None

    def delete_source(self, source_id: str) -> bool:
        result = self._call_rpc(self.RPC_DELETE_SOURCE, [[[source_id]], [2]])
        return result is not None

    def get_source_guide(self, source_id: str) -> dict[str, Any]:
        result = self._call_rpc(self.RPC_GET_SOURCE_GUIDE, [[[[source_id]]]], "/")
        summary, keywords = "", []
        if result and isinstance(result, list):
            if len(result) > 0 and isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], list):
                inner = result[0][0]
                if len(inner) > 1 and isinstance(inner[1], list) and len(inner[1]) > 0:
                    summary = inner[1][0]
                if len(inner) > 2 and isinstance(inner[2], list) and len(inner[2]) > 0:
                    keywords = inner[2][0] if isinstance(inner[2][0], list) else []
        return {"summary": summary, "keywords": keywords}

    def get_source_fulltext(self, source_id: str) -> dict[str, Any]:
        params = [[source_id], [2], [2]]
        result = self._call_rpc(self.RPC_GET_SOURCE, params, "/")
        content, title, source_type, url = "", "", "", None
        if result and isinstance(result, list):
            if len(result) > 0 and isinstance(result[0], list):
                sm = result[0]
                if len(sm) > 1 and isinstance(sm[1], str):
                    title = sm[1]
                if len(sm) > 2 and isinstance(sm[2], list):
                    meta = sm[2]
                    if len(meta) > 4:
                        source_type = constants.SOURCE_TYPES.get_name(meta[4])
                    if len(meta) > 7 and isinstance(meta[7], list) and len(meta[7]) > 0:
                        url = meta[7][0]
            if len(result) > 3 and isinstance(result[3], list) and len(result[3]) > 0 and isinstance(result[3][0], list):
                parts = []
                for block in result[3][0]:
                    if isinstance(block, list):
                        parts.extend(self._extract_all_text(block))
                content = "\n\n".join(parts)
        return {"content": content, "title": title, "source_type": source_type, "url": url, "char_count": len(content)}

    def wait_for_source_ready(self, notebook_id: str, source_id: str,
                              timeout: float = 120.0, poll_interval: float = 3.0) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            sources = self.get_notebook_sources_with_types(notebook_id)
            for src in sources:
                if src.get("id") == source_id:
                    if src.get("status") == self.SOURCE_STATUS_READY:
                        return src
                    if src.get("status") == self.SOURCE_STATUS_ERROR:
                        raise RuntimeError(f"Source {source_id} failed to process")
                    break
            time.sleep(poll_interval)
        raise TimeoutError(f"Source {source_id} not ready after {timeout}s")

    # =====================================================================
    # Conversation / Query
    # =====================================================================

    def query(self, notebook_id: str, query_text: str, source_ids: list[str] | None = None,
              conversation_id: str | None = None, timeout: float = QUERY_TIMEOUT) -> dict | None:
        client = self._get_client()
        if source_ids is None:
            nb_data = self.get_notebook(notebook_id)
            source_ids = self._extract_source_ids(nb_data)

        is_new = conversation_id is None
        if is_new:
            conversation_id = str(uuid.uuid4())
            history = None
        else:
            history = self._build_conversation_history(conversation_id)

        sources_array = [[[sid]] for sid in source_ids] if source_ids else []
        params = [sources_array, query_text, history, [2, None, [1]], conversation_id]
        params_json = json.dumps(params, separators=(",", ":"))
        f_req_json = json.dumps([None, params_json], separators=(",", ":"))
        parts = [f"f.req={urllib.parse.quote(f_req_json, safe='')}"]
        if self.csrf_token:
            parts.append(f"at={urllib.parse.quote(self.csrf_token, safe='')}")
        body = "&".join(parts) + "&"

        self._reqid_counter += 100000
        url_params = {
            "bl": os.environ.get("NOTEBOOKLM_BL", "boq_labs-tailwind-frontend_20260108.06_p0"),
            "hl": "en", "_reqid": str(self._reqid_counter), "rt": "c",
        }
        if self._session_id:
            url_params["f.sid"] = self._session_id
        url = f"{self.BASE_URL}{self.QUERY_ENDPOINT}?{urllib.parse.urlencode(url_params)}"

        response = client.post(url, content=body, timeout=timeout)
        response.raise_for_status()
        answer = self._parse_query_response(response.text)
        if answer:
            self._cache_turn(conversation_id, query_text, answer)
        turns = self._conversation_cache.get(conversation_id, [])
        return {
            "answer": answer, "conversation_id": conversation_id,
            "turn_number": len(turns), "is_follow_up": not is_new,
        }

    def clear_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self._conversation_cache:
            del self._conversation_cache[conversation_id]
            return True
        return False

    # =====================================================================
    # Studio operations
    # =====================================================================

    def _get_all_source_ids(self, notebook_id: str) -> list[str]:
        try:
            return [s["id"] for s in self.get_notebook_sources_with_types(notebook_id) if s.get("id")]
        except Exception:
            return []

    def create_audio_overview(self, notebook_id: str, source_ids: list[str] | None = None,
                              format_code: int = 1, length_code: int = 2,
                              language: str = "en", focus_prompt: str = "") -> dict | None:
        if source_ids is None:
            source_ids = self._get_all_source_ids(notebook_id)
        if not source_ids:
            raise ValueError(f"No sources in notebook {notebook_id}")
        sn = [[[sid]] for sid in source_ids]
        ss = [[sid] for sid in source_ids]
        opts = [None, [focus_prompt, length_code, None, ss, language, None, format_code]]
        params = [[2], notebook_id, [None, None, constants.STUDIO_TYPE_AUDIO, sn, None, None, opts]]
        return self._parse_studio_create(
            self._call_rpc(self.RPC_CREATE_STUDIO, params, f"/notebook/{notebook_id}"),
            notebook_id, "audio",
        )

    def create_video_overview(self, notebook_id: str, source_ids: list[str] | None = None,
                              format_code: int = 1, visual_style_code: int = 1,
                              language: str = "en", focus_prompt: str = "") -> dict | None:
        if source_ids is None:
            source_ids = self._get_all_source_ids(notebook_id)
        if not source_ids:
            raise ValueError(f"No sources in notebook {notebook_id}")
        sn = [[[sid]] for sid in source_ids]
        ss = [[sid] for sid in source_ids]
        opts = [None, None, [ss, language, focus_prompt, None, format_code, visual_style_code]]
        params = [[2], notebook_id, [None, None, constants.STUDIO_TYPE_VIDEO, sn, None, None, None, None, opts]]
        return self._parse_studio_create(
            self._call_rpc(self.RPC_CREATE_STUDIO, params, f"/notebook/{notebook_id}"),
            notebook_id, "video",
        )

    def create_report(self, notebook_id: str, source_ids: list[str] | None = None,
                      report_format: str = "Briefing Doc", custom_prompt: str = "",
                      language: str = "en") -> dict | None:
        if source_ids is None:
            source_ids = self._get_all_source_ids(notebook_id)
        if not source_ids:
            raise ValueError(f"No sources in notebook {notebook_id}")
        configs = {
            "Briefing Doc": ("Briefing Doc", "Key insights and important quotes",
                             "Create a comprehensive briefing document with Executive Summary, themes, quotes, and insights."),
            "Study Guide": ("Study Guide", "Short-answer quiz, essay questions, glossary",
                            "Create a study guide with key concepts, practice questions, essay prompts, and glossary."),
            "Blog Post": ("Blog Post", "Insightful takeaways in readable article format",
                          "Write an engaging blog post with introduction, sections, and takeaways."),
            "Create Your Own": ("Custom Report", "Custom format",
                                custom_prompt or "Create a report based on the provided sources."),
        }
        if report_format not in configs:
            raise ValueError(f"Invalid report_format: {report_format}. Options: {list(configs.keys())}")
        t, d, p = configs[report_format]
        sn = [[[sid]] for sid in source_ids]
        ss = [[sid] for sid in source_ids]
        opts = [None, [t, d, None, ss, language, p, None, True]]
        params = [[2], notebook_id, [None, None, constants.STUDIO_TYPE_REPORT, sn, None, None, None, opts]]
        return self._parse_studio_create(
            self._call_rpc(self.RPC_CREATE_STUDIO, params, f"/notebook/{notebook_id}"),
            notebook_id, "report",
        )

    def create_flashcards(self, notebook_id: str, source_ids: list[str] | None = None,
                          difficulty_code: int = 2) -> dict | None:
        if source_ids is None:
            source_ids = self._get_all_source_ids(notebook_id)
        if not source_ids:
            raise ValueError(f"No sources in notebook {notebook_id}")
        sn = [[[sid]] for sid in source_ids]
        opts = [None, [1, None, None, None, None, None, [difficulty_code, constants.FLASHCARD_COUNT_DEFAULT]]]
        params = [[2], notebook_id, [None, None, constants.STUDIO_TYPE_FLASHCARDS, sn, None, None, None, None, None, opts]]
        return self._parse_studio_create(
            self._call_rpc(self.RPC_CREATE_STUDIO, params, f"/notebook/{notebook_id}"),
            notebook_id, "flashcards",
        )

    def create_quiz(self, notebook_id: str, source_ids: list[str] | None = None,
                    question_count: int = 2, difficulty: int = 2) -> dict | None:
        if source_ids is None:
            source_ids = self._get_all_source_ids(notebook_id)
        if not source_ids:
            raise ValueError(f"No sources in notebook {notebook_id}")
        sn = [[[sid]] for sid in source_ids]
        opts = [None, [2, None, None, None, None, None, None, [question_count, difficulty]]]
        params = [[2], notebook_id, [None, None, constants.STUDIO_TYPE_FLASHCARDS, sn, None, None, None, None, None, opts]]
        return self._parse_studio_create(
            self._call_rpc(self.RPC_CREATE_STUDIO, params, f"/notebook/{notebook_id}"),
            notebook_id, "quiz",
        )

    def create_infographic(self, notebook_id: str, source_ids: list[str] | None = None,
                           orientation_code: int = 1, detail_level_code: int = 2,
                           language: str = "en", focus_prompt: str = "") -> dict | None:
        if source_ids is None:
            source_ids = self._get_all_source_ids(notebook_id)
        if not source_ids:
            raise ValueError(f"No sources in notebook {notebook_id}")
        sn = [[[sid]] for sid in source_ids]
        opts = [[focus_prompt or None, language, None, orientation_code, detail_level_code]]
        content = [None, None, constants.STUDIO_TYPE_INFOGRAPHIC, sn] + [None] * 10 + [opts]
        params = [[2], notebook_id, content]
        return self._parse_studio_create(
            self._call_rpc(self.RPC_CREATE_STUDIO, params, f"/notebook/{notebook_id}"),
            notebook_id, "infographic",
        )

    def create_slide_deck(self, notebook_id: str, source_ids: list[str] | None = None,
                          format_code: int = 1, length_code: int = 3,
                          language: str = "en", focus_prompt: str = "") -> dict | None:
        if source_ids is None:
            source_ids = self._get_all_source_ids(notebook_id)
        if not source_ids:
            raise ValueError(f"No sources in notebook {notebook_id}")
        sn = [[[sid]] for sid in source_ids]
        opts = [[focus_prompt or None, language, format_code, length_code]]
        content = [None, None, constants.STUDIO_TYPE_SLIDE_DECK, sn] + [None] * 12 + [opts]
        params = [[2], notebook_id, content]
        return self._parse_studio_create(
            self._call_rpc(self.RPC_CREATE_STUDIO, params, f"/notebook/{notebook_id}"),
            notebook_id, "slide_deck",
        )

    def create_data_table(self, notebook_id: str, source_ids: list[str] | None = None,
                          description: str = "", language: str = "en") -> dict | None:
        if source_ids is None:
            source_ids = self._get_all_source_ids(notebook_id)
        if not source_ids:
            raise ValueError(f"No sources in notebook {notebook_id}")
        sn = [[[sid]] for sid in source_ids]
        opts = [None, [description, language]]
        content = [None, None, constants.STUDIO_TYPE_DATA_TABLE, sn] + [None] * 14 + [opts]
        params = [[2], notebook_id, content]
        return self._parse_studio_create(
            self._call_rpc(self.RPC_CREATE_STUDIO, params, f"/notebook/{notebook_id}"),
            notebook_id, "data_table",
        )

    def poll_studio_status(self, notebook_id: str) -> list[dict]:
        params = [[2], notebook_id, 'NOT artifact.status = "ARTIFACT_STATUS_SUGGESTED"']
        result = self._call_rpc(self.RPC_POLL_STUDIO, params, f"/notebook/{notebook_id}")
        artifacts = []
        if not result or not isinstance(result, list) or len(result) == 0:
            return artifacts
        alist = result[0] if isinstance(result[0], list) else result
        type_map = {
            constants.STUDIO_TYPE_AUDIO: "audio", constants.STUDIO_TYPE_REPORT: "report",
            constants.STUDIO_TYPE_VIDEO: "video", constants.STUDIO_TYPE_FLASHCARDS: "flashcards",
            constants.STUDIO_TYPE_INFOGRAPHIC: "infographic", constants.STUDIO_TYPE_SLIDE_DECK: "slide_deck",
            constants.STUDIO_TYPE_DATA_TABLE: "data_table",
        }
        for ad in alist:
            if not isinstance(ad, list) or len(ad) < 5:
                continue
            aid = ad[0]
            title = ad[1] if len(ad) > 1 else ""
            tc = ad[2] if len(ad) > 2 else None
            sc = ad[4] if len(ad) > 4 else None
            is_quiz = False
            if tc == constants.STUDIO_TYPE_FLASHCARDS and len(ad) > 9:
                fo = ad[9]
                if isinstance(fo, list) and len(fo) > 1 and isinstance(fo[1], list) and len(fo[1]) > 0:
                    if fo[1][0] == 2:
                        is_quiz = True
            atype = "quiz" if is_quiz else type_map.get(tc, "unknown")
            status = "in_progress" if sc == 1 else "completed" if sc == 3 else "unknown"
            artifacts.append({"artifact_id": aid, "title": title, "type": atype, "status": status})
        return artifacts

    def delete_studio_artifact(self, artifact_id: str, notebook_id: str | None = None) -> bool:
        try:
            result = self._call_rpc(self.RPC_DELETE_STUDIO, [[2], artifact_id])
            if result is not None:
                return True
        except Exception:
            pass
        if notebook_id:
            return self.delete_mind_map(notebook_id, artifact_id)
        return False

    # =====================================================================
    # Mind maps
    # =====================================================================

    def generate_mind_map(self, notebook_id: str, source_ids: list[str] | None = None) -> dict | None:
        if source_ids is None:
            source_ids = self._get_all_source_ids(notebook_id)
        if not source_ids:
            raise ValueError(f"No sources in notebook {notebook_id}")
        sn = [[[sid]] for sid in source_ids]
        params = [sn, None, None, None, None, ["interactive_mindmap", [["[CONTEXT]", ""]], ""], None, [2, None, [1]]]
        result = self._call_rpc(self.RPC_GENERATE_MIND_MAP, params)
        if result and isinstance(result, list) and len(result) > 0:
            inner = result[0] if isinstance(result[0], list) else result
            mm_json = inner[0] if isinstance(inner[0], str) else None
            gen_id = None
            if len(inner) > 2 and isinstance(inner[2], list) and len(inner[2]) > 0:
                gen_id = inner[2][0]
            return {"mind_map_json": mm_json, "generation_id": gen_id, "source_ids": source_ids}
        return None

    def save_mind_map(self, notebook_id: str, mind_map_json: str, source_ids: list[str] | None = None,
                      title: str = "Mind Map") -> dict | None:
        if source_ids is None:
            source_ids = self._get_all_source_ids(notebook_id)
        if not source_ids:
            raise ValueError(f"No sources in notebook {notebook_id}")
        ss = [[sid] for sid in source_ids]
        meta = [2, None, None, 5, ss]
        params = [notebook_id, mind_map_json, meta, None, title]
        result = self._call_rpc(self.RPC_SAVE_MIND_MAP, params, f"/notebook/{notebook_id}")
        if result and isinstance(result, list) and len(result) > 0:
            inner = result[0] if isinstance(result[0], list) else result
            return {"mind_map_id": inner[0] if len(inner) > 0 else None, "notebook_id": notebook_id, "title": title}
        return None

    def list_mind_maps(self, notebook_id: str) -> list[dict]:
        result = self._call_rpc(self.RPC_LIST_MIND_MAPS, [notebook_id], f"/notebook/{notebook_id}")
        maps = []
        if result and isinstance(result, list) and len(result) > 0:
            for entry in (result[0] if isinstance(result[0], list) else []):
                if not isinstance(entry, list) or len(entry) < 2 or entry[1] is None:
                    continue
                details = entry[1]
                if isinstance(details, list) and len(details) >= 5:
                    maps.append({
                        "mind_map_id": entry[0], "title": details[4] if len(details) > 4 else "Mind Map",
                        "mind_map_json": details[1] if len(details) > 1 else None,
                    })
        return maps

    def delete_mind_map(self, notebook_id: str, mind_map_id: str) -> bool:
        params = [notebook_id, None, [mind_map_id], [2]]
        self._call_rpc(self.RPC_DELETE_MIND_MAP, params, f"/notebook/{notebook_id}")
        return True

    # =====================================================================
    # Research
    # =====================================================================

    def start_research(self, notebook_id: str, query: str, source: str = "web",
                       mode: str = "fast") -> dict | None:
        sl, ml = source.lower(), mode.lower()
        if sl not in ("web", "drive"):
            raise ValueError(f"Invalid source '{source}'. Use 'web' or 'drive'.")
        if ml not in ("fast", "deep"):
            raise ValueError(f"Invalid mode '{mode}'. Use 'fast' or 'deep'.")
        if ml == "deep" and sl == "drive":
            raise ValueError("Deep Research only supports Web sources.")
        st = constants.RESEARCH_SOURCE_WEB if sl == "web" else constants.RESEARCH_SOURCE_DRIVE
        if ml == "fast":
            params = [[query, st], None, 1, notebook_id]
            rpc = self.RPC_START_FAST_RESEARCH
        else:
            params = [None, [1], [query, st], 5, notebook_id]
            rpc = self.RPC_START_DEEP_RESEARCH
        result = self._call_rpc(rpc, params, f"/notebook/{notebook_id}")
        if result and isinstance(result, list) and len(result) > 0:
            return {"task_id": result[0], "report_id": result[1] if len(result) > 1 else None,
                    "notebook_id": notebook_id, "query": query, "source": sl, "mode": ml}
        return None

    def poll_research(self, notebook_id: str, target_task_id: str | None = None) -> dict | None:
        result = self._call_rpc(self.RPC_POLL_RESEARCH, [None, None, notebook_id], f"/notebook/{notebook_id}")
        if not result or not isinstance(result, list) or len(result) == 0:
            return {"status": "no_research", "message": "No active research found"}
        if isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], list):
            result = result[0]
        for td in result:
            if not isinstance(td, list) or len(td) < 2 or not isinstance(td[0], str):
                continue
            tid = td[0]
            ti = td[1] if len(td) > 1 and isinstance(td[1], list) else None
            if not ti:
                continue
            if target_task_id and tid != target_task_id:
                continue
            qi = ti[1] if len(ti) > 1 else None
            sc = ti[4] if len(ti) > 4 else None
            sas = ti[3] if len(ti) > 3 and isinstance(ti[3], list) else []
            sources_data = sas[0] if sas and isinstance(sas[0], list) else []
            summary = sas[1] if len(sas) >= 2 and isinstance(sas[1], str) else ""
            sources = self._parse_research_sources(sources_data)
            return {
                "task_id": tid, "status": "completed" if sc in (2, 6) else "in_progress",
                "query": qi[0] if qi and len(qi) > 0 else "",
                "source_type": "web" if (qi and len(qi) > 1 and qi[1] == 1) else "drive",
                "sources": sources, "source_count": len(sources), "summary": summary,
            }
        return {"status": "no_research", "message": "No active research found"}

    def import_research_sources(self, notebook_id: str, task_id: str, sources: list[dict]) -> list[dict]:
        if not sources:
            return []
        sa = []
        for src in sources:
            url, title, rt = src.get("url", ""), src.get("title", "Untitled"), src.get("result_type", 1)
            if rt == 5 or not url:
                continue
            if rt == 1:
                sa.append([None, None, [url, title], None, None, None, None, None, None, None, 2])
            else:
                doc_id = url.split("id=")[-1].split("&")[0] if "id=" in url else None
                if doc_id:
                    mt = {2: "application/vnd.google-apps.document", 3: "application/vnd.google-apps.presentation",
                          8: "application/vnd.google-apps.spreadsheet"}.get(rt, "application/vnd.google-apps.document")
                    sa.append([[doc_id, mt, 1, title], None, None, None, None, None, None, None, None, None, 2])
                else:
                    sa.append([None, None, [url, title], None, None, None, None, None, None, None, 2])
        params = [None, [1], task_id, notebook_id, sa]
        result = self._call_rpc(self.RPC_IMPORT_RESEARCH, params, f"/notebook/{notebook_id}", timeout=120.0)
        imported = []
        if result and isinstance(result, list):
            if len(result) > 0 and isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], list):
                result = result[0]
            for sd in result:
                if isinstance(sd, list) and len(sd) >= 2:
                    sid = sd[0][0] if sd[0] and isinstance(sd[0], list) else None
                    if sid:
                        imported.append({"id": sid, "title": sd[1] if len(sd) > 1 else "Untitled"})
        return imported

    # =====================================================================
    # Sharing
    # =====================================================================

    def get_share_status(self, notebook_id: str) -> ShareStatus:
        result = self._call_rpc(self.RPC_GET_SHARE_STATUS, [notebook_id, [2]])
        collabs: list[Collaborator] = []
        is_public = False
        if result and isinstance(result, list):
            for item in result:
                if isinstance(item, list):
                    for entry in item:
                        if isinstance(entry, list) and len(entry) >= 2:
                            email = entry[0] if entry[0] and isinstance(entry[0], str) and "@" in entry[0] else None
                            if email:
                                rc = entry[1] if len(entry) > 1 and isinstance(entry[1], int) else 3
                                dn = None
                                if len(entry) > 3 and isinstance(entry[3], list) and len(entry[3]) > 0:
                                    dn = str(entry[3][0])
                                ip = len(entry) > 4 and entry[4] is True
                                collabs.append(Collaborator(
                                    email=email, role=constants.SHARE_ROLES.get_name(rc),
                                    is_pending=ip, display_name=dn))
            for item in result:
                if isinstance(item, list) and len(item) >= 1 and item[0] == 1:
                    is_public = True
                    break
        link = f"https://notebooklm.google.com/notebook/{notebook_id}" if is_public else None
        return ShareStatus(is_public=is_public, access_level="public" if is_public else "restricted",
                           collaborators=collabs, public_link=link)

    def set_public_access(self, notebook_id: str, is_public: bool = True) -> str | None:
        ac = constants.SHARE_ACCESS_PUBLIC if is_public else constants.SHARE_ACCESS_RESTRICTED
        params = [[[notebook_id, None, [ac], [0, ""]]], 1, None, [2]]
        self._call_rpc(self.RPC_SHARE_NOTEBOOK, params)
        return f"https://notebooklm.google.com/notebook/{notebook_id}" if is_public else None

    def add_collaborator(self, notebook_id: str, email: str, role: str = "viewer",
                         notify: bool = True, message: str = "") -> bool:
        rc = constants.SHARE_ROLES.get_code(role)
        if rc == constants.SHARE_ROLE_OWNER:
            raise ValueError("Cannot add collaborator as owner")
        nf = 0 if notify else 1
        params = [[[notebook_id, [[email, None, rc]], None, [nf, message]]], 1, None, [2]]
        result = self._call_rpc(self.RPC_SHARE_NOTEBOOK, params)
        return result is not None

    # =====================================================================
    # Notes
    # =====================================================================

    def create_note(self, notebook_id: str, content: str, title: str | None = None) -> dict | None:
        if title is None:
            title = "New Note"
        params = [notebook_id, "", [1], None, title]
        result = self._call_rpc(self.RPC_CREATE_NOTE, params, f"/notebook/{notebook_id}")
        if result and isinstance(result, list) and len(result) > 0:
            nd = result[0] if isinstance(result[0], list) else result
            nid = nd[0] if isinstance(nd, list) and len(nd) > 0 else nd
            if nid and content:
                self.update_note(nid, content=content, title=title, notebook_id=notebook_id)
                return {"id": nid, "title": title, "content": content}
            return {"id": nid, "title": title, "content": ""}
        return None

    def list_notes(self, notebook_id: str) -> list[dict]:
        result = self._call_rpc(self.RPC_GET_NOTES, [notebook_id], f"/notebook/{notebook_id}")
        notes = []
        if result and isinstance(result, list) and len(result) > 0:
            items = result[0] if isinstance(result[0], list) else []
            for item in items:
                if not isinstance(item, list) or len(item) < 2:
                    continue
                if (len(item) > 2 and item[2] == 2) or item[1] is None:
                    continue
                nd = item[1] if isinstance(item[1], list) else None
                if nd and len(nd) >= 5:
                    content = nd[1] if len(nd) > 1 else ""
                    title = nd[4] if len(nd) > 4 else "Untitled"
                    # Skip mind maps (JSON with children/nodes)
                    if content:
                        try:
                            parsed = json.loads(content)
                            if isinstance(parsed, dict) and ("children" in parsed or "nodes" in parsed):
                                continue
                        except (json.JSONDecodeError, TypeError):
                            pass
                    notes.append({"id": item[0], "title": title, "content": content, "preview": content[:100] if content else ""})
        return notes

    def update_note(self, note_id: str, content: str | None = None, title: str | None = None,
                    notebook_id: str | None = None) -> dict | None:
        if not notebook_id:
            raise ValueError("notebook_id is required")
        if content is None and title is None:
            raise ValueError("Must provide content or title")
        if content is not None and title is not None:
            nc, nt = content, title
        else:
            all_notes = self.list_notes(notebook_id)
            cur = next((n for n in all_notes if n["id"] == note_id), None)
            if not cur:
                return None
            nc = content if content is not None else cur.get("content", "")
            nt = title if title is not None else cur.get("title", "")
        params = [notebook_id, note_id, [[[nc, nt, [], 0]]]]
        self._call_rpc(self.RPC_UPDATE_NOTE, params, f"/notebook/{notebook_id}")
        return {"id": note_id, "title": nt, "content": nc}

    def delete_note(self, note_id: str, notebook_id: str) -> bool:
        self._call_rpc(self.RPC_DELETE_NOTE, [notebook_id, None, [note_id]], f"/notebook/{notebook_id}")
        return True

    # =====================================================================
    # Export
    # =====================================================================

    def export_artifact(self, notebook_id: str, artifact_id: str, title: str = "NotebookLM Export",
                        export_type: str = "docs", content: str | None = None) -> dict[str, Any]:
        etc = constants.EXPORT_TYPES.get_code(export_type)
        params = [None, artifact_id, content, title, etc]
        result = self._call_rpc(self.RPC_EXPORT_ARTIFACT, params, f"/notebook/{notebook_id}")
        doc_url = None
        if result and isinstance(result, list):
            if len(result) > 0 and isinstance(result[0], list):
                if len(result[0]) > 0 and isinstance(result[0][0], list) and result[0][0]:
                    doc_url = result[0][0][0]
                elif len(result[0]) > 0 and isinstance(result[0][0], str):
                    doc_url = result[0][0]
            elif len(result) > 0 and isinstance(result[0], str):
                doc_url = result[0]
        if doc_url:
            return {"status": "success", "url": doc_url, "message": f"Exported to: {doc_url}"}
        return {"status": "failed", "url": None, "message": "Export failed"}

    # =====================================================================
    # Private helpers
    # =====================================================================

    def _parse_source_add_result(self, result: Any) -> dict | None:
        if result and isinstance(result, list) and len(result) > 0:
            sl = result[0] if result else []
            if sl and len(sl) > 0:
                sd = sl[0]
                sid = sd[0][0] if isinstance(sd, list) and sd[0] else None
                title = sd[1] if isinstance(sd, list) and len(sd) > 1 else "Untitled"
                return {"id": sid, "title": title}
        return None

    def _extract_nested_id(self, data: Any) -> str | None:
        if isinstance(data, str):
            return data
        if isinstance(data, list) and len(data) > 0:
            return self._extract_nested_id(data[0])
        return None

    def _extract_all_text(self, data: list) -> list[str]:
        texts = []
        for item in data:
            if isinstance(item, str) and len(item) > 0:
                texts.append(item)
            elif isinstance(item, list):
                texts.extend(self._extract_all_text(item))
        return texts

    def _extract_source_ids(self, notebook_data: Any) -> list[str]:
        ids = []
        if not notebook_data or not isinstance(notebook_data, list):
            return ids
        try:
            if len(notebook_data) > 0 and isinstance(notebook_data[0], list):
                ni = notebook_data[0]
                if len(ni) > 1 and isinstance(ni[1], list):
                    for src in ni[1]:
                        if isinstance(src, list) and len(src) > 0:
                            sw = src[0]
                            if isinstance(sw, list) and len(sw) > 0 and isinstance(sw[0], str):
                                ids.append(sw[0])
        except (IndexError, TypeError):
            pass
        return ids

    def _build_conversation_history(self, conversation_id: str) -> list | None:
        turns = self._conversation_cache.get(conversation_id, [])
        if not turns:
            return None
        history = []
        for t in turns:
            history.append([t.answer, None, 2])
            history.append([t.query, None, 1])
        return history or None

    def _cache_turn(self, cid: str, query: str, answer: str) -> None:
        if cid not in self._conversation_cache:
            self._conversation_cache[cid] = []
        n = len(self._conversation_cache[cid]) + 1
        self._conversation_cache[cid].append(ConversationTurn(query=query, answer=answer, turn_number=n))

    def _parse_query_response(self, text: str) -> str:
        if text.startswith(")]}'"):
            text = text[4:]
        lines = text.strip().split("\n")
        longest_answer = ""
        longest_thinking = ""
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            try:
                int(line)
                i += 1
                if i < len(lines):
                    t, ia = self._extract_answer_chunk(lines[i])
                    if t:
                        if ia and len(t) > len(longest_answer):
                            longest_answer = t
                        elif not ia and len(t) > len(longest_thinking):
                            longest_thinking = t
                i += 1
            except ValueError:
                t, ia = self._extract_answer_chunk(line)
                if t:
                    if ia and len(t) > len(longest_answer):
                        longest_answer = t
                    elif not ia and len(t) > len(longest_thinking):
                        longest_thinking = t
                i += 1
        return longest_answer if longest_answer else longest_thinking

    def _extract_answer_chunk(self, json_str: str) -> tuple[str | None, bool]:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None, False
        if not isinstance(data, list):
            return None, False
        for item in data:
            if not isinstance(item, list) or len(item) < 3 or item[0] != "wrb.fr":
                continue
            inner_str = item[2]
            if not isinstance(inner_str, str):
                continue
            try:
                inner = json.loads(inner_str)
            except json.JSONDecodeError:
                continue
            if isinstance(inner, list) and len(inner) > 0:
                fe = inner[0]
                if isinstance(fe, list) and len(fe) > 0:
                    at = fe[0]
                    if isinstance(at, str) and len(at) > 20:
                        is_answer = False
                        if len(fe) > 4 and isinstance(fe[4], list) and len(fe[4]) > 0:
                            if isinstance(fe[4][-1], int):
                                is_answer = fe[4][-1] == 1
                        return at, is_answer
                elif isinstance(fe, str) and len(fe) > 20:
                    return fe, False
        return None, False

    def _parse_studio_create(self, result: Any, notebook_id: str, atype: str) -> dict | None:
        if result and isinstance(result, list) and len(result) > 0:
            ad = result[0]
            aid = ad[0] if isinstance(ad, list) and len(ad) > 0 else None
            sc = ad[4] if isinstance(ad, list) and len(ad) > 4 else None
            return {
                "artifact_id": aid, "notebook_id": notebook_id, "type": atype,
                "status": "in_progress" if sc == 1 else "completed" if sc == 3 else "unknown",
            }
        return None

    def _parse_research_sources(self, sources_data: list) -> list[dict]:
        sources = []
        for idx, src in enumerate(sources_data):
            if not isinstance(src, list) or len(src) < 2:
                continue
            if src[0] is None and len(src) > 1 and isinstance(src[1], str):
                rt = src[3] if len(src) > 3 and isinstance(src[3], int) else 5
                sources.append({"index": idx, "url": "", "title": src[1], "description": "",
                                "result_type": rt, "result_type_name": constants.RESULT_TYPES.get_name(rt)})
            elif isinstance(src[0], str) or len(src) >= 3:
                url = src[0] if isinstance(src[0], str) else ""
                title = src[1] if len(src) > 1 and isinstance(src[1], str) else ""
                desc = src[2] if len(src) > 2 and isinstance(src[2], str) else ""
                rt = src[3] if len(src) > 3 and isinstance(src[3], int) else 1
                sources.append({"index": idx, "url": url, "title": title, "description": desc,
                                "result_type": rt, "result_type_name": constants.RESULT_TYPES.get_name(rt)})
        return sources

    def _start_resumable_upload(self, notebook_id: str, filename: str, file_size: int, source_id: str) -> str:
        url = f"{self.UPLOAD_URL}?authuser=0"
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Origin": self.BASE_URL, "Referer": f"{self.BASE_URL}/",
            "x-goog-authuser": "0", "x-goog-upload-command": "start",
            "x-goog-upload-header-content-length": str(file_size),
            "x-goog-upload-protocol": "resumable",
        }
        body = json.dumps({"PROJECT_ID": notebook_id, "SOURCE_NAME": filename, "SOURCE_ID": source_id})
        with httpx.Client(timeout=60.0, cookies=self._get_httpx_cookies()) as client:
            response = client.post(url, headers=headers, content=body)
            response.raise_for_status()
            upload_url = response.headers.get("x-goog-upload-url")
            if not upload_url:
                raise RuntimeError("Failed to get upload URL")
            return upload_url

    def _upload_file_streaming(self, upload_url: str, file_path: Path) -> None:
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
            "Origin": self.BASE_URL, "Referer": f"{self.BASE_URL}/",
            "x-goog-authuser": "0", "x-goog-upload-command": "upload, finalize",
            "x-goog-upload-offset": "0",
        }
        def stream():
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk
        with httpx.Client(timeout=300.0, cookies=self._get_httpx_cookies()) as client:
            response = client.post(upload_url, headers=headers, content=stream())
            response.raise_for_status()
