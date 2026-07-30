#!/usr/bin/env python3
"""
Wiring audit — find guarded imports that reference symbols which do not exist.

The failure mode this catches
-----------------------------
This codebase guards optional integrations with try/except ImportError and
degrades to a fallback. That is a good pattern for genuinely optional third-party
dependencies. Applied to *repo-local* imports it is a trap: when a refactor
renames or moves a symbol, the ImportError is swallowed and the feature silently
turns itself off. Nothing fails, nothing logs, and the capability is simply gone.

Three instances were found by hand before this script existed:

  mcp_raw/tools/react_synthesis.py  imported a CoherenceDetector that does not
      exist, so every ReACT run executed 3 of its 4 tools while reporting 4.

  delegation/four_ds.py  imported LLMRequest from cpb.llm_client, which does not
      define it, so HAS_LLM_CLIENT was False on every run and the LLM-enhanced
      4Ds description gate was permanently unreachable.

  storage/ucw_ingestion.py and scripts/context-packs/build_packs.py  imported
      run_oracle_consensus from the critic package, which never re-exported it,
      so oracle consensus silently never ran in pack building or UCW ingestion.

Scope
-----
Repo-local imports only. Third-party guards (rich, qdrant_client, PIL, ...) are
legitimately optional and are ignored — this script cannot and should not decide
whether an optional dependency is installed.

Usage
-----
  python3 scripts/audit/wiring_audit.py           # report, exit 0
  python3 scripts/audit/wiring_audit.py --ci      # report, exit 1 on findings
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]

# Vendored, generated, or archived trees — not our wiring.
SKIP_PARTS = {
    ".git",
    ".graveyard",
    ".venv",
    ".beads",
    ".ralph-tui",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "dashboard",
    "frontier-alpha-cvrf",
    "visual_assets",
}


def is_skipped(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def python_files() -> List[Path]:
    return [f for f in ROOT.rglob("*.py") if not is_skipped(f.relative_to(ROOT))]


def module_defines(path: Path) -> Optional[Set[str]]:
    """
    Top-level names a module provides.

    Walks conditional bodies (try/except, if/else) because availability flags and
    fallback definitions in this codebase are routinely defined inside them —
    e.g. `except ImportError: QDRANT_AVAILABLE = False`. Missing those produces
    false positives, which would make the audit untrustworthy and get it ignored.
    """
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except (SyntaxError, UnicodeDecodeError):
        return None

    names: Set[str] = set()

    def collect(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for el in target.elts:
                            if isinstance(el, ast.Name):
                                names.add(el.id)
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                names.add(child.target.id)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    names.add(alias.asname or alias.name.split(".")[0])

            # Descend only through control flow, not into function bodies.
            if isinstance(child, (ast.Try, ast.If, ast.ExceptHandler, ast.With)):
                collect(child)

    collect(tree)
    return names


def build_module_index() -> Tuple[Dict[str, Path], Set[str]]:
    """
    Map importable module names to files, plus the set of package names.

    Registers both the dotted path from the repo root (`storage.qdrant_db`) and
    the bare stem (`qdrant_db`). The bare form matters because several entry
    points insert their own directory onto sys.path and then import siblings
    unqualified.
    """
    modules: Dict[str, Path] = {}
    packages: Set[str] = set()

    for f in python_files():
        rel = f.relative_to(ROOT)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
            if parts:
                packages.add(".".join(parts))
                packages.add(parts[-1])
        if not parts:
            continue
        modules.setdefault(".".join(parts), f)
        modules.setdefault(parts[-1], f)

    return modules, packages


def resolve(module: str, importer: Path, modules: Dict[str, Path]) -> Optional[Path]:
    """
    Resolve a module name to a repo file, preferring the importer's own package.

    `from embeddings import _get_model` inside mcp_raw/server.py means
    mcp_raw/embeddings.py, not some other embeddings.py elsewhere in the tree.
    Resolving purely by bare name picks whichever file happened to be indexed
    first and invents failures that do not exist.
    """
    sibling = importer.parent / f"{module.replace('.', '/')}.py"
    if sibling.exists():
        return sibling

    sibling_pkg = importer.parent / module.replace(".", "/") / "__init__.py"
    if sibling_pkg.exists():
        return sibling_pkg

    return modules.get(module)


def audit() -> List[Tuple[str, int, str, str, str]]:
    modules, packages = build_module_index()
    findings: List[Tuple[str, int, str, str, str]] = []

    for f in python_files():
        rel = f.relative_to(ROOT)
        try:
            tree = ast.parse(f.read_text(errors="ignore"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue

            for stmt in ast.walk(node):
                if not isinstance(stmt, ast.ImportFrom) or not stmt.module:
                    continue

                target = resolve(stmt.module, f, modules)
                if target is None:
                    continue  # third-party or unresolvable — out of scope

                defined = module_defines(target)
                if defined is None:
                    continue

                target_rel = target.relative_to(ROOT)
                is_package = target.name == "__init__.py"

                for alias in stmt.names:
                    if alias.name == "*":
                        continue
                    if alias.name in defined:
                        continue

                    # `from pkg import submodule` is valid even when the package
                    # __init__ never names it.
                    if is_package:
                        sub = target.parent / f"{alias.name}.py"
                        subpkg = target.parent / alias.name / "__init__.py"
                        if sub.exists() or subpkg.exists():
                            continue

                    findings.append(
                        (
                            str(rel),
                            stmt.lineno,
                            stmt.module,
                            alias.name,
                            str(target_rel),
                        )
                    )

    return findings


# Accepted, tracked gaps. Keyed "path::module::symbol" — deliberately without
# line numbers so unrelated edits above them do not churn this list.
#
# Everything here is a real silent failure that needs a decision rather than a
# mechanical fix. The point of the baseline is that CI can gate on *new*
# regressions today instead of waiting for the backlog to clear — a check that
# cannot fail is the antipattern this audit exists to correct.
# Empty. Both entries were the orphaned critic/oracle_adapter.py; the design
# decision they were waiting on has been made — see .graveyard/MANIFEST.md.
BASELINE: Dict[str, str] = {}


def key(path: str, module: str, symbol: str) -> str:
    return f"{path}::{module}::{symbol}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci",
        action="store_true",
        help="exit non-zero on findings outside the tracked baseline",
    )
    args = parser.parse_args()

    findings = audit()
    new = [f for f in findings if key(f[0], f[2], f[3]) not in BASELINE]
    known = [f for f in findings if key(f[0], f[2], f[3]) in BASELINE]

    if known:
        print(f"ℹ️  wiring audit: {len(known)} known gap(s) in the tracked baseline")
        for path, lineno, module, symbol, _ in known:
            print(f"     {path}:{lineno}  from {module} import {symbol}")
        print()

    if not new:
        print("✅ wiring audit: no new phantom repo-local guarded imports")
        return 0

    print(f"❌ wiring audit: {len(new)} new phantom repo-local guarded import(s)\n")
    print("These imports are caught by a try/except and degrade silently — the")
    print("feature behind each one is off, and nothing reports it.\n")
    for path, lineno, module, symbol, target in new:
        print(f"  {path}:{lineno}")
        print(f"      from {module} import {symbol}")
        print(f"      {target} does not define {symbol}\n")

    return 1 if args.ci else 0


if __name__ == "__main__":
    sys.exit(main())
