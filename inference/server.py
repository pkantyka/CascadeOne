"""One local System One HTTP endpoint backed by a configured CascadeOne cascade.

Uses standard library only and reuses cascades.runtime for routing. Backend model
services (fast and strong) must already be listening at the configured URLs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cascades"))
from runtime import Cascade
from inference.report import read_report, startup_lines


MAX_BODY = 2 * 1024 * 1024


class BadRequest(ValueError):
    pass


def get_json(url, body=None, timeout=30):
    encoded = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=encoded, headers={"Content-Type": "application/json"} if encoded is not None else {}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read(MAX_BODY + 1))


def check_backend(url, expected, timeout=10):
    info = get_json(url.rstrip("/") + "/v1/models", timeout=timeout)
    models = info.get("models", [])
    if not isinstance(models, list) or len(models) != 1:
        raise RuntimeError("expected exactly one model at " + url)
    model = models[0]
    if model.get("name") != expected["model_id"].split("/")[-1]:
        raise RuntimeError("backend model identity mismatch at " + url)
    for field, value in (("revision", expected.get("revision")),
                         ("mode", expected.get("mode")),
                         ("orders", expected.get("orders"))):
        if value is not None and field in model and model[field] != value:
            raise RuntimeError(f"backend {field} mismatch at {url}")
    if "revision" not in model:
        logging.warning("Backend %s did not attest revision; config pin is not verified by HTTP", url)
    return model


def check_request(payload, model_id):
    if not isinstance(payload, dict):
        raise BadRequest("request must be a JSON object")
    model = payload.get("model", model_id)
    if model not in (model_id, "cascadeone"):
        raise BadRequest("model must identify this cascade")
    if "state" not in payload or not isinstance(payload["state"], (str, dict, list)):
        raise BadRequest("state must be a string, object or array")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 32:
        raise BadRequest("questions must contain 1-32 entries")
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid or not isinstance(q, dict):
            raise BadRequest("invalid question")
        kind = q.get("type")
        if kind == "choice":
            if not isinstance(q.get("criteria"), dict) or not q["criteria"]:
                raise BadRequest("choice requires options")
        elif kind == "score":
            if not isinstance(q.get("criteria"), list) or len(q["criteria"]) < 2:
                raise BadRequest("score requires at least two levels")
        elif kind != "noul":
            raise BadRequest("unsupported question type")
    return payload


def check_prediction(request, result, *, fast):
    if fast and result.get("fast_eligible") is False:
        return
    if not isinstance(result.get("answers"), dict) or set(result["answers"]) != set(request["questions"]):
        raise ValueError("upstream answer set does not match request")
    for qid, q in request["questions"].items():
        answer = result["answers"][qid]
        if not isinstance(answer, dict) or not isinstance(answer.get("probabilities"), dict):
            raise ValueError("upstream answer missing probability distribution")
        ids = (list(q["criteria"]) if q["type"] == "choice"
               else [str(i) for i in range(len(q["criteria"]))] if q["type"] == "score"
               else ["false", "true"])
        probs = answer["probabilities"]
        if set(probs) != set(ids):
            raise ValueError("upstream probability options mismatch")
        if any(type(v) not in (int, float) or not 0 <= v <= 1 for v in probs.values()):
            raise ValueError("invalid upstream probability")
        if not 0.998 <= sum(probs.values()) <= 1.002:
            raise ValueError("upstream probabilities do not sum to one")


class Service:
    def __init__(self, cascade_path, benchmark=None, timeout=360,
                 min_speedup_pct=10.0, max_quality_loss_pp=2.0, max_p95_ratio=1.10):
        path = Path(cascade_path).resolve()
        self.config = json.loads(path.read_text(encoding="utf-8"))
        if self.config.get("schema") != "cascadeone-cascade-v0.1":
            raise ValueError("unsupported cascade schema")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.cascade = Cascade(self.config)
        self.timeout = timeout
        self.benchmark = None
        if benchmark:
            self.benchmark = read_report(
                path, self.config, benchmark,
                min_speedup_pct=min_speedup_pct,
                max_quality_loss_pp=max_quality_loss_pp,
                max_p95_ratio=max_p95_ratio
            )
        self.startup = startup_lines(self.benchmark)

    def ensure_backends(self, wait_seconds=600):
        deadline = time.monotonic() + wait_seconds
        for role in ("fast", "strong"):
            cfg = self.config[role]
            announced = False
            while True:
                try:
                    check_backend(cfg["url"], cfg)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"{role} model unavailable at {cfg['url']}") from err
                    if not announced:
                        logging.info("[CascadeOne] Waiting for %s backend at %s", role, cfg["url"])
                        announced = True
                    time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))

    def infer(self, payload):
        check_request(payload, self.config["id"])

        def invoke(url, request, timeout, fast=False):
            # 'model' is the facade's ID, not the backend's model name.
            forwarded = dict(request)
            forwarded.pop("model", None)
            started = time.perf_counter_ns()
            result = get_json(url.rstrip("/") + "/v1/systemone", forwarded, timeout)
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            check_prediction(request, result, fast=fast)
            return result, elapsed

        outcome = self.cascade.execute(payload, self.timeout, invoke)
        return outcome


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CascadeOneInference/0.1"

        def send_json(self, status, payload, extra_headers=None):
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self.send_json(200, {"status": "ok", "cascade": service.config["id"]})
            elif path == "/v1/models":
                self.send_json(200, {"models": [{
                    "name": service.config["id"],
                    "mode": "cascade",
                    "orders": 1
                }]})
            else:
                self.send_json(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/v1/systemone":
                self.send_json(404, {"error": "unknown endpoint"})
                return
            try:
                raw_length = self.headers.get("Content-Length")
                if raw_length is None or not raw_length.isdigit():
                    raise BadRequest("Content-Length required")
                size = int(raw_length)
                if size < 1 or size > MAX_BODY:
                    raise BadRequest("request body empty or exceeds 2 MiB")
                try:
                    payload = json.loads(self.rfile.read(size))
                except (UnicodeDecodeError, json.JSONDecodeError) as err:
                    raise BadRequest("invalid JSON") from err
                outcome = service.infer(payload)
                response = dict(outcome.response)
                response.pop("fast_eligible", None)
                self.send_json(200, response, {
                    "X-CascadeOne-Route": outcome.route["reason"],
                    "X-CascadeOne-Escalated": str(outcome.escalated).lower(),
                    "X-CascadeOne-Policy": outcome.route["policy"]
                })
            except BadRequest as err:
                self.send_json(400, {"error": str(err)})
            except urllib.error.HTTPError as err:
                logging.warning("Backend returned HTTP %d", err.code)
                self.send_json(502, {"error": "upstream model service failed"})
            except (urllib.error.URLError, TimeoutError) as err:
                logging.warning("Backend unavailable: %s", err)
                self.send_json(502, {"error": "upstream model service unavailable"})
            except Exception:
                logging.exception("Cascade inference failed")
                self.send_json(502, {"error": "cascade inference failed"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cascade", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path,
                        help="completed, validated colocated benchmark run directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18358)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--backend-wait", type=int, default=600)
    parser.add_argument("--min-speedup-pct", type=float, default=10.0)
    parser.add_argument("--max-quality-loss-pp", type=float, default=2.0)
    parser.add_argument("--max-p95-ratio", type=float, default=1.10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run = args.benchmark or os.environ.get("CASCADEONE_BENCHMARK_RUN") or None
    service = Service(args.cascade, benchmark=run, timeout=args.timeout,
                      min_speedup_pct=args.min_speedup_pct,
                      max_quality_loss_pp=args.max_quality_loss_pp,
                      max_p95_ratio=args.max_p95_ratio)
    for line in service.startup:
        logging.warning(line) if "[WARN]" in line else logging.info(line)
    if args.backend_wait < 0:
        parser.error("--backend-wait must be non-negative")
    service.ensure_backends(wait_seconds=args.backend_wait)
    with ThreadingHTTPServer((args.host, args.port), make_handler(service)) as http:
        logging.info("[CascadeOne] Serving %s on http://%s:%d",
                     service.config["id"], args.host, args.port)
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            logging.warning("[CascadeOne][WARN] Inference endpoint is not restricted to loopback; use trusted networking.")
        http.serve_forever()


if __name__ == "__main__":
    main()
