"""Loopback HTTP and real process boundaries, with isolated fixture storage."""

import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from basix.handoff import validate_handoff
from basix.planner import validate_receipt
from basix.store import DecisionStore
from basix.web import MAX_BODY, make_server


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def interface_bundle(tmp_path):
    now = datetime.now(timezone.utc)
    scope = {"width": 64, "height": 64, "count": 1, "scale": 4}
    content = "Synthetic interface fixture, not a real performance claim."
    bundle = {
        "workload": {"id": "interface-test", "task": "image_upscale", "name": "Interface fixture",
                     "max_cost_usd": 1, "deadline_seconds": 30, "cloud_upload_allowed": False,
                     "allowed_regions": [], "input": scope},
        "offers": [{"id": "fixture", "name": "Fixture candidate", "provider": "Test only", "task": "image_upscale",
                    "region": "local", "data_leaves_device": False, "availability": "available",
                    "estimated_seconds": 1, "estimated_cost_usd": 0, "measurement_scope": scope,
                    "evidence_ids": ["fixture-evidence"], "notes": "No execution."}],
        "evidence": [{"id": "fixture-evidence", "title": "Interface fixture", "uri": "urn:basix:test:interfaces",
                      "captured_at": (now - timedelta(minutes=1)).isoformat(), "valid_until": (now + timedelta(hours=1)).isoformat(),
                      "kind": "synthetic_fixture", "content": content, "sha256": hashlib.sha256(content.encode()).hexdigest()}],
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    return bundle, path


@pytest.fixture
def live_http(tmp_path, interface_bundle):
    bundle, path = interface_bundle
    store = DecisionStore(tmp_path / "http-receipts")
    server = make_server(store, path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, store, bundle
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def nonce_for(server):
    status, headers, body = request(server, "GET", "/")
    assert status == 200
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    return re.search(rb'<meta name="basix-nonce" content="([^"]+)"', body).group(1).decode()


def test_http_evaluate_restore_and_handoff(live_http):
    server, store, bundle = live_http
    assert server.server_address[0] == "127.0.0.1"
    nonce = nonce_for(server)
    status, headers, payload = request(server, "POST", "/api/evaluate", json.dumps(bundle["workload"]),
                                       {"Content-Type": "application/json", "X-Basix-Nonce": nonce,
                                        "Origin": f"http://127.0.0.1:{server.server_port}"})
    assert status == 200
    receipt = validate_receipt(json.loads(payload))
    assert receipt["recommended_offer_id"] is None  # A fixture cannot prove feasibility.
    assert receipt["execution_authorized"] is False
    assert store.get(receipt["decision_id"]) == receipt
    assert json.loads(request(server, "GET", "/api/decisions")[2]) == [receipt]
    assert json.loads(request(server, "GET", f"/api/decisions/{receipt['decision_id']}")[2]) == receipt
    status, headers, markdown = request(server, "GET", f"/api/decisions/{receipt['decision_id']}/handoff")
    assert status == 200
    assert headers["Content-Type"].startswith("text/markdown")
    assert receipt["decision_id"].encode() in markdown
    assert b"not granted" in markdown


@pytest.mark.parametrize("extra_headers", [
    {"Host": "evil.example"}, {"Origin": "https://evil.example"},
    {"Origin": "null"}, {"X-Basix-Nonce": "bad"}, {"X-Basix-Nonce": "é"},
])
def test_http_rejects_foreign_origins_hosts_and_wrong_nonce(live_http, extra_headers):
    server, store, bundle = live_http
    headers = {"Content-Type": "application/json", "X-Basix-Nonce": nonce_for(server)}
    headers.update(extra_headers)
    assert request(server, "POST", "/api/evaluate", json.dumps(bundle["workload"]), headers)[0] == 403
    assert store.list() == []
    if "Host" in extra_headers or "Origin" in extra_headers:
        assert request(server, "GET", "/api/bundle", headers=extra_headers)[0] == 403


def test_http_validation_and_size_errors_do_not_persist(live_http):
    server, store, bundle = live_http
    headers = {"Content-Type": "application/json", "X-Basix-Nonce": nonce_for(server)}
    assert request(server, "POST", "/api/evaluate", "{}", {"Content-Type": "application/json"})[0] == 403
    assert request(server, "POST", "/api/evaluate", "{}", {**headers, "Content-Type": "text/plain"})[0] == 415
    for body in ("not JSON", "null", "[]", "{}"):
        assert request(server, "POST", "/api/evaluate", body, headers)[0] == 400
    assert request(server, "POST", "/api/evaluate", "", {**headers, "Content-Length": str(MAX_BODY + 1)})[0] == 413
    assert request(server, "POST", "/api/evaluate", "{}", {**headers, "Content-Length": "nonsense"})[0] == 400
    assert request(server, "GET", "/api/decisions/not-a-hash")[0] == 404
    assert store.list() == []


def installed_python():
    candidate = ROOT / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def process_env():
    # The test must prove an installed import outside the checkout, not succeed
    # because an inherited PYTHONPATH incidentally points at the source tree.
    return {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}


def cli(tmp_path, *args, input_text=None):
    return subprocess.run([installed_python(), "-m", "basix", "--state-dir", str(tmp_path / "process-receipts"), *args],
                          cwd=tmp_path, env=process_env(), input=input_text, text=True,
                          capture_output=True, timeout=20)


def test_installed_cli_from_foreign_cwd_and_complete_round_trip(tmp_path, interface_bundle):
    _, path = interface_bundle
    process = cli(tmp_path, "evaluate", str(path))
    assert process.returncode == 0, process.stderr
    receipt = validate_receipt(json.loads(process.stdout))
    decision_id = receipt["decision_id"]
    assert json.loads(cli(tmp_path, "get", decision_id).stdout) == receipt
    assert json.loads(cli(tmp_path, "list").stdout) == [receipt]
    handoff = cli(tmp_path, "handoff", decision_id)
    assert handoff.returncode == 0, handoff.stderr
    assert decision_id in handoff.stdout
    assert "not granted" in handoff.stdout
    probe = subprocess.run([installed_python(), "-c",
                            "from importlib.resources import files; import basix; "
                            "assert files('basix').joinpath('static/index.html').is_file(); "
                            "assert files('basix').joinpath('examples/media-upscale.json').is_file(); print(basix.__file__)"],
                           cwd=tmp_path, env=process_env(), text=True, capture_output=True, timeout=10)
    assert probe.returncode == 0, probe.stderr
    assert str(ROOT) in probe.stdout


def test_cli_invalid_bundle_cannot_create_receipt(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"workload": null}')
    process = cli(tmp_path, "evaluate", str(path))
    assert process.returncode != 0
    assert DecisionStore(tmp_path / "process-receipts").list() == []


def test_mcp_stdio_initialize_tools_restore_and_export(tmp_path, interface_bundle):
    bundle, path = interface_bundle
    saved = cli(tmp_path, "evaluate", str(path))
    assert saved.returncode == 0, saved.stderr
    original = json.loads(saved.stdout)
    decision_id = original["decision_id"]
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": {"name": "test-client", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "basix_evaluate_workload", "arguments": bundle}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "basix_get_decision", "arguments": {"decision_id": decision_id}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "basix_export_handoff", "arguments": {"decision_id": decision_id}}},
    ]
    process = cli(tmp_path, "mcp", input_text="".join(json.dumps(item) + "\n" for item in messages))
    assert process.returncode == 0, process.stderr
    responses = [json.loads(line) for line in process.stdout.splitlines()]
    assert [item["id"] for item in responses] == [1, 2, 3, 4, 5]
    assert responses[0]["result"]["serverInfo"]["name"] == "basix-researchgravity"
    assert {tool["name"] for tool in responses[1]["result"]["tools"]} == {"basix_evaluate_workload", "basix_get_decision", "basix_export_handoff"}
    assert validate_receipt(json.loads(responses[2]["result"]["content"][0]["text"]))["execution_authorized"] is False
    assert json.loads(responses[3]["result"]["content"][0]["text"]) == original
    assert validate_handoff(json.loads(responses[4]["result"]["content"][0]["text"]))["decision_id"] == decision_id
    assert (tmp_path / "process-receipts" / "mcp-logs" / "mcp.log").exists()


def test_mcp_malformed_arguments_and_tool_notifications_do_not_write(tmp_path, interface_bundle):
    bundle, _ = interface_bundle
    messages = [
        {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "basix_evaluate_workload", "arguments": bundle}},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": None},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "basix_evaluate_workload", "arguments": []}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "basix_evaluate_workload", "arguments": {**bundle, "execute": True}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "basix_get_decision", "arguments": {"decision_id": "../../private"}}},
    ]
    process = cli(tmp_path, "mcp", input_text="{" + "\n" + "".join(json.dumps(item) + "\n" for item in messages))
    assert process.returncode == 0, process.stderr
    responses = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(responses) == 5
    assert responses[0]["error"]["code"] == -32700
    assert responses[1]["error"]["code"] == -32602
    assert all(item["result"]["isError"] for item in responses[2:])
    assert DecisionStore(tmp_path / "process-receipts").list() == []


def test_mcp_overlimit_line_errors_and_exits_cleanly(tmp_path):
    process = cli(tmp_path, "mcp", input_text=" " * 512_001 + "\n")
    assert process.returncode == 0, process.stderr
    output = json.loads(process.stdout)
    assert output["error"]["code"] == -32600
    assert DecisionStore(tmp_path / "process-receipts").list() == []


def test_protocol_import_has_no_capture_or_logger_side_effects(tmp_path):
    probe = subprocess.run(
        [installed_python(), "-c",
         "import sys; from mcp_raw.protocol import validate_message; "
         "from mcp_raw import Config; "
         "assert 'mcp_raw.server' not in sys.modules; "
         "assert 'mcp_raw.capture' not in sys.modules; "
         "assert 'mcp_raw.logger' not in sys.modules; "
         "assert validate_message({'jsonrpc':'2.0','id':1,'method':'ping'}) == 'request'"],
        cwd=tmp_path, env=process_env(), text=True, capture_output=True, timeout=10)
    assert probe.returncode == 0, probe.stderr
