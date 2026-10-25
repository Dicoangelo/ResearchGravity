"""Loopback-only planner UI. Every mutation uses this server's session nonce."""
import hmac
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .handoff import export_handoff, render_handoff
from .planner import evaluate_workload

MAX_BODY = 512_000


def make_server(store, bundle_path: Path, port: int = 8097):
    bundle = json.loads(bundle_path.read_text())
    nonce = secrets.token_urlsafe(32)
    template = (Path(__file__).parent / "static" / "index.html").read_text()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # No request bodies, paths, or user content in logs.

        def trusted(self):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in hosts:
                return False
            origin = self.headers.get("Origin")
            return origin is None or origin in {"http://" + h for h in hosts}

        def respond(self, status, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, (str, bytes)):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False)
            payload = body.encode() if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if not self.trusted():
                return self.respond(403, {"error": "Local origin required"})
            route = urlsplit(self.path).path
            try:
                if route == "/":
                    return self.respond(200, template.replace("__NONCE__", nonce), "text/html; charset=utf-8")
                if route == "/api/bundle":
                    return self.respond(200, bundle)
                if route == "/api/decisions":
                    return self.respond(200, store.list())
                if route.startswith("/api/decisions/"):
                    parts = route.split("/")
                    receipt = store.get(parts[3])
                    if len(parts) == 5 and parts[4] == "handoff":
                        return self.respond(200, render_handoff(export_handoff(receipt)), "text/markdown; charset=utf-8")
                    if len(parts) == 4:
                        return self.respond(200, receipt)
                return self.respond(404, {"error": "Not found"})
            except (ValueError, FileNotFoundError):
                return self.respond(404, {"error": "Receipt unavailable or invalid"})

        def do_POST(self):
            if not self.trusted() or not hmac.compare_digest(self.headers.get("X-Basix-Nonce", "").encode("utf-8"), nonce.encode("utf-8")):
                return self.respond(403, {"error": "Local session required"})
            if self.path != "/api/evaluate":
                return self.respond(404, {"error": "Not found"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.respond(415, {"error": "JSON required"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    return self.respond(413, {"error": "Request exceeds size limit"})
                workload = json.loads(self.rfile.read(length))
                receipt = evaluate_workload(workload, bundle["offers"], bundle["evidence"])
                store.save(receipt)
                return self.respond(200, receipt)
            except (ValueError, KeyError, TypeError) as exc:
                return self.respond(400, {"error": str(exc)[:250]})
            except OSError:
                return self.respond(503, {"error": "Decision could not be saved; no success recorded"})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve_local(store, bundle_path: Path, port: int):
    server = make_server(store, bundle_path, port)
    print(f"BASIX local workspace: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
