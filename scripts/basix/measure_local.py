#!/usr/bin/env python3
"""One bounded local measurement using the existing Sovereign Media image tool.

Creates its own 64x64 PNG, runs only the installed local Upscayl lane, and records
dimensions, hashes and elapsed wall time. This is not a visual-quality benchmark.
No credentials, network, cloud provisioning or private input are used.
"""
import argparse
import hashlib
import json
import os
import signal
import struct
import subprocess
import time
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path


def run_bounded(command, *, timeout=90):
    """Run a POSIX process group and stop the whole group if interrupted.

    The selected local tool is a shell wrapper. Killing only that shell would
    leave its GPU worker running, so the shell and descendants share a new
    session that can be terminated together. No provider interface is invoked.
    """
    if os.name != "posix":
        raise RuntimeError("Bounded local measurements require POSIX process groups")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    try:
        output, errors = process.communicate(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The entire group already exited.
        finally:
            process.communicate()
        raise
    return subprocess.CompletedProcess(command, process.returncode, output, errors)


def png_bytes(width=64, height=64):
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))
    raw = b"".join(b"\0" + bytes(v for x in range(width) for v in
        (x * 255 // width, y * 255 // height, 220 if (x // 8 + y // 8) % 2 else 40))
        for y in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tool", type=Path, default=Path.home() / "projects/sovereign-media/bin/upscale-image")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "sample-64.png"
    dest = args.output_dir / "sample-256.png"
    if source.exists() or dest.exists():
        raise SystemExit("Use a new output directory; measurements never overwrite artifacts")
    source.write_bytes(png_bytes())
    start = time.monotonic()
    try:
        result = run_bounded([str(args.tool), str(source), "--scale", "4", "--model", "upscayl-lite-4x", "--out", str(dest)])
    except subprocess.TimeoutExpired:
        raise SystemExit("Local upscale timed out; process group stopped; no successful measurement emitted")
    elapsed = time.monotonic() - start
    if result.returncode or not dest.exists():
        raise SystemExit("Local upscale failed; no successful measurement emitted")
    output = dest.read_bytes()
    if output[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", output[16:24]) != (256, 256):
        raise SystemExit("Output shape failed; no successful measurement emitted")
    now = datetime.now(timezone.utc)
    content = json.dumps({"engine": "Real-ESRGAN / Upscayl upscayl-lite-4x", "wall_seconds": round(elapsed, 3),
        "input_dimensions": [64, 64], "output_dimensions": [256, 256], "count": 1, "scale": 4,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "output_sha256": hashlib.sha256(output).hexdigest(),
        "validation": "PNG signature and 4x dimensions only; perceptual quality and repeatability untested",
        "provider_charge_usd": 0, "cost_scope": "Incremental provider charge only; energy/depreciation not measured"}, sort_keys=True)
    evidence = {"id": "local-upscale-measurement", "title": "Observed local 64x64 image upscale", "uri": "urn:basix:artifact:sample-256.png",
        "captured_at": now.isoformat(), "valid_until": (now + timedelta(days=1)).isoformat(), "kind": "local_measurement",
        "content": content, "sha256": hashlib.sha256(content.encode()).hexdigest()}
    historical = "Sovereign Media README dated 2026-07-12 records a RunPod RTX 4090 upscale at 2.07 seconds/frame, 768x1168, $0.69/hour. This is a historical report, not a current quote or benchmark for 64x64 images."
    old = {"id": "historical-runpod", "title": "Archived Sovereign Media GPU observation", "uri": "urn:basix:source:sovereign-media:README.md:ac2971b",
        "captured_at": "2026-07-12T12:00:00+00:00", "valid_until": "2026-07-13T12:00:00+00:00", "kind": "historical_record",
        "content": historical, "sha256": hashlib.sha256(historical.encode()).hexdigest()}
    scope = {"width": 64, "height": 64, "count": 1, "scale": 4}
    bundle = {"workload": {"id": "local-upscale-pilot", "task": "image_upscale", "name": "Upscale one calibration image 4x",
        "max_cost_usd": 1, "deadline_seconds": 30, "cloud_upload_allowed": False, "allowed_regions": [], "input": scope},
        "offers": [{"id": "local-upscayl", "name": "Local Upscayl", "provider": "Sovereign Media", "task": "image_upscale",
            "region": "local", "data_leaves_device": False, "availability": "available", "estimated_seconds": round(elapsed, 3),
            "estimated_cost_usd": 0, "measurement_scope": scope, "evidence_ids": [evidence["id"]],
            "notes": "Single calibration run. Runtime varies by image content. Incremental provider charge is zero; energy and hardware cost are excluded. Output quality has not been scored."},
            {"id": "runpod-historical", "name": "RunPod GPU lane", "provider": "RunPod", "task": "image_upscale",
            "region": None, "data_leaves_device": True, "availability": "unknown", "estimated_seconds": None,
            "estimated_cost_usd": None, "measurement_scope": None, "evidence_ids": [old["id"]],
            "notes": "Historical lane only. Current capacity, region, all-in cost, and this workload's performance have not been measured. Cloud executor disabled."}],
        "evidence": [evidence, old]}
    (args.output_dir / "media-upscale.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(json.dumps({"measurement": "passed", "wall_seconds": round(elapsed, 3), "output_dimensions": [256, 256], "bundle": str(args.output_dir / "media-upscale.json")}))


if __name__ == "__main__":
    main()
