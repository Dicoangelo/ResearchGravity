"""Explicit, local entry point; no provider discovery or automatic execution."""
import argparse
import asyncio
import json
from pathlib import Path

from .handoff import export_handoff, render_handoff
from .planner import evaluate_workload
from .store import DecisionStore


def main():
    parser = argparse.ArgumentParser(description="BASIX evidence-backed workload planner")
    parser.add_argument("--state-dir", type=Path, required=True,
                        help="Dedicated local receipt directory; never your private UCW corpus")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("bundle", type=Path)
    sub.add_parser("list")
    for name in ("get", "handoff", "index"):
        cmd = sub.add_parser(name)
        cmd.add_argument("decision_id")
        if name == "index":
            cmd.add_argument("--db", type=Path, required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8097)
    serve.add_argument("--bundle", type=Path,
                       default=Path(__file__).parent / "examples" / "media-upscale.json")
    sub.add_parser("mcp")
    args = parser.parse_args()
    store = DecisionStore(args.state_dir)
    if args.command == "evaluate":
        bundle = json.loads(args.bundle.read_text())
        receipt = evaluate_workload(bundle["workload"], bundle["offers"], bundle["evidence"])
        store.save(receipt)
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
    elif args.command == "list":
        print(json.dumps(store.list(), indent=2, ensure_ascii=False))
    elif args.command == "get":
        print(json.dumps(store.get(args.decision_id), indent=2, ensure_ascii=False))
    elif args.command == "handoff":
        print(render_handoff(export_handoff(store.get(args.decision_id))))
    elif args.command == "index":
        from .researchgravity import index_receipt
        print(json.dumps(asyncio.run(index_receipt(store.get(args.decision_id), args.db)), indent=2))
    elif args.command == "serve":
        from .web import serve_local
        serve_local(store, args.bundle, args.port)
    elif args.command == "mcp":
        from .mcp import serve_stdio
        asyncio.run(serve_stdio(store))


if __name__ == "__main__":
    main()
