"""Immutable, local decision receipts; no ambient ResearchGravity state."""

import json
import os
import re
import tempfile
from pathlib import Path

from .planner import validate_receipt


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RECEIPT_BYTES = 8 * 1024 * 1024


def _checked_id(decision_id: str) -> str:
    if not isinstance(decision_id, str) or not _DIGEST.fullmatch(decision_id):
        raise ValueError("decision_id must be a lowercase SHA-256 digest")
    return decision_id


def _read_receipt(path: Path, expected_id: str) -> dict:
    if path.is_symlink():
        raise ValueError("Receipt symlinks are not accepted")
    if path.stat().st_size > _MAX_RECEIPT_BYTES:
        raise ValueError("Receipt exceeds the 8 MiB limit")
    try:
        receipt = validate_receipt(json.loads(path.read_text(encoding="utf-8")))
    except (UnicodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError(f"Invalid receipt: {path.name}") from exc
    if receipt["decision_id"] != expected_id:
        raise ValueError("Receipt content does not match its filename")
    return receipt


class DecisionStore:
    """Content-addressed JSON store with atomic, exclusive publication.

    The fully written temporary file is hard-linked to the destination. Unlike
    check-then-replace, this never overwrites an existing receipt, including when
    two processes publish simultaneously. Existing corrupt data is an error.
    """

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    def save(self, receipt: dict) -> Path:
        checked = validate_receipt(receipt)
        decision_id = _checked_id(checked["decision_id"])
        payload = (json.dumps(checked, sort_keys=True, ensure_ascii=False,
                              allow_nan=False, indent=2) + "\n").encode("utf-8")
        if len(payload) > _MAX_RECEIPT_BYTES:
            raise ValueError("Receipt exceeds the 8 MiB limit")
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{decision_id}.json"
        fd, temporary_name = tempfile.mkstemp(prefix=".receipt-", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                existing = _read_receipt(target, decision_id)
                if existing != checked:
                    raise ValueError("Conflicting receipt already exists")
            else:
                # Persist the directory entry as well as the file content.
                directory_fd = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def get(self, decision_id: str) -> dict:
        checked_id = _checked_id(decision_id)
        return _read_receipt(self.root / f"{checked_id}.json", checked_id)

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        receipts = [self.get(path.stem) for path in self.root.glob("*.json")]
        return sorted(receipts, key=lambda item: (item["created_at"], item["decision_id"]),
                      reverse=True)
