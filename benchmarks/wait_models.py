"""Wait for Pico, Ultra, and the matching CascadeOne inference service."""
import argparse
import json
from pathlib import Path
import time
from runner import check_identity, request_json, sha

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cascades/basal-pico-ultra/benchmark.json"

def check_ready(inference_url):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for role in ("fast", "strong"):
        endpoint = config[role]["url"].rstrip("/")
        request_json(endpoint + "/health", timeout=15)
        check_identity(endpoint, config[role])
    info = request_json(inference_url.rstrip("/") + "/v1/models", timeout=15)
    models = info.get("models", [])
    if len(models) != 1 or models[0].get("name") != config["id"]:
        raise ValueError("inference cascade identity mismatch")
    if models[0].get("cascade_config_sha256") != sha(CONFIG):
        raise ValueError("inference cascade configuration mismatch")
    if models[0].get("residency") != "colocated":
        raise ValueError("inference is not in colocated mode")
    attestations = models[0].get("backend_attestations", {})
    if not {"fast", "strong"} <= set(attestations):
        raise ValueError("inference backend identity checks are incomplete")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--inference-url", default="http://inference:18358")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        try:
            check_ready(args.inference_url)
            print("PICO_ULTRA_INFERENCE_READY", flush=True)
            return
        except (OSError, TimeoutError, RuntimeError, ValueError) as exc:
            print("Waiting for pinned backends and inference service:", str(exc), flush=True)
            time.sleep(min(10, max(0, deadline - time.monotonic())))
    raise TimeoutError("pinned backends and inference service were not ready before timeout")

if __name__ == "__main__":
    main()
