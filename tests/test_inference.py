"""CPU-only contract tests for the local single-model cascade facade."""
from __future__ import annotations

import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.server import Service, make_handler, BadRequest
from inference.report import read_report, startup_lines


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {
            "schema": "cascadeone-cascade-v0.1",
            "id": "mock-cascade",
            "fast": {"url": "http://fast:18357", "model_id": "demo/fast", "revision": "f"},
            "strong": {"url": "http://strong:18357", "model_id": "demo/strong", "revision": "s"},
            "routing": {"type": "confidence_threshold", "parameters": {"threshold": 0.7}},
        }
        self.config_path = self.root / "cascade.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.request = {"model": "mock-cascade", "state": "Customer", "questions": {
            "q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}
        }}

    def backend(self, confidence):
        calls = []
        def fake(url, body=None, timeout=30):
            calls.append((url, body))
            if url.endswith("/v1/models"):
                role = "strong" if "strong" in url else "fast"
                return {"models": [{"name": role, "revision": self.config[role]["revision"]}]}
            value = confidence if "fast" in url else 0.99
            return {"answers": {"q": {"probabilities": {"a": value, "b": 1-value},
                                     "confidence": value}}}
        return calls, fake

    def write_report(self, mean_cascade=70, quality_cascade=.90, quality_strong=.92,
                     p95_cascade=100, include_gold=True):
        root = self.root / "run"
        root.mkdir(exist_ok=True)
        manifest = {
            "schema": "cascadeone-run-v0.1", "status": "complete",
            "cascade_id": self.config["id"],
            "cascade_config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            "phase": "colocated", "repeats": 1, "n_requests": 2,
            "dataset_sha256": "0" * 64,
            "dataset_identity": {"mode": "local_unregistered", "id": "benchmark-user"},
            "normalized_requests_sha256": "a"*64,
            "models": {"fast": self.config["fast"], "strong": self.config["strong"]},
            "routing": self.config["routing"],
            "benchmark_setup": {"id": "test-benchmark", "adapter": "typed_decisions_v1", "selection": {"type": "all"}}
        }
        arms = {
            "fast_only": {"n": 2, "unique_requests": 2, "mean_ms": 40, "p95_ms": 50},
            "strong_only": {"n": 2, "unique_requests": 2, "mean_ms": 100, "p95_ms": 110},
            "cascade": {"n": 2, "unique_requests": 2, "mean_ms": mean_cascade, "p95_ms": p95_cascade},
        }
        arms["strong_only"]["question_expected_accuracy"] = quality_strong if include_gold else None
        arms["cascade"]["question_expected_accuracy"] = quality_cascade if include_gold else None
        summary = {"schema": "cascadeone-summary-v0.1", "arms": arms}
        rows = []
        for sample in ("1", "2"):
            for arm in ("fast_only", "strong_only", "cascade"):
                escalated = arm != "fast_only"
                rows.append({"id": sample, "repeat": 0, "arm": arm,
                             "escalated": escalated,
                             "routing": {"policy": self.config["routing"]["type"] if arm == "cascade" else arm,
                                         "escalated": escalated, "signals": {}}})
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "summary.json").write_text(json.dumps(summary))
        (root / "environment.json").write_text("{}")
        (root / "measurements.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        (root / "telemetry.jsonl").write_text("{}\n")
        files = sorted(p for p in root.iterdir() if p.is_file())
        (root / "SHA256SUMS").write_text("".join(
            hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
            for p in files
        ))
        return root

    def test_report_correct_speedup_and_quality(self):
        path = self.write_report()
        report = read_report(self.config_path, self.config, path)
        self.assertAlmostEqual(report["latency_reduction_pct"], 30)
        self.assertAlmostEqual(report["speedup_factor"], 100/70)
        self.assertAlmostEqual(report["quality"]["delta_pp"], -2)
        self.assertFalse(any("Quality loss" in w for w in report["warnings"]))
        self.assertIn("Gold-based question accuracy", "\n".join(startup_lines(report)))

    def test_report_alerts_for_slow_and_low_quality(self):
        path = self.write_report(mean_cascade=110, quality_cascade=.75,
                                 p95_cascade=150)
        report = read_report(self.config_path, self.config, path)
        msg = "\n".join(report["warnings"])
        self.assertIn("Quality loss", msg)
        self.assertIn("Mean latency reduction", msg)
        self.assertIn("p95 latency ratio", msg)
        self.assertLess(report["latency_reduction_pct"], 0)

    def test_unknown_quality_and_missing_report(self):
        self.assertIn("UNKNOWN", "\n".join(startup_lines(None)))
        path = self.write_report(include_gold=False)
        report = read_report(self.config_path, self.config, path)
        self.assertIsNone(report["quality"])
        self.assertIn("Gold quality unavailable", "\n".join(report["warnings"]))

    def test_rejects_mismatched_or_tampered_benchmark(self):
        path = self.write_report()
        different = dict(self.config, routing={"type": "always_fast"})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            read_report(self.config_path, different, path)
        (path / "summary.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            read_report(self.config_path, self.config, path)

    def test_server_routing_and_same_api_model(self):
        service = Service(self.config_path)
        calls, fake = self.backend(.95)
        with mock.patch("inference.server.get_json", side_effect=fake):
            ok = service.infer(self.request)
            self.assertFalse(ok.escalated)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("model", calls[0][1])
            calls.clear()
            low_calls, low = self.backend(.6)
            with mock.patch("inference.server.get_json", side_effect=low):
                fallback = service.infer(self.request)
            self.assertTrue(fallback.escalated)
            self.assertEqual(len(low_calls), 2)

    def test_http_systemone_and_models_endpoint(self):
        service = Service(self.config_path)
        calls, fake = self.backend(.95)
        http = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        url = "http://127.0.0.1:" + str(http.server_port)
        with mock.patch("inference.server.get_json", side_effect=fake):
            with urllib.request.urlopen(url+"/v1/models") as response:
                models = json.load(response)["models"]
            self.assertEqual(models[0]["name"], "mock-cascade")
            payload = json.dumps(self.request).encode()
            req = urllib.request.Request(url+"/v1/systemone", data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as response:
                self.assertEqual(response.headers["X-CascadeOne-Escalated"], "false")
                result = json.load(response)
            self.assertEqual(set(result["answers"]), {"q"})
            self.assertEqual(len(calls), 1)

    def test_client_validation(self):
        service = Service(self.config_path)
        with self.assertRaises(BadRequest):
            service.infer({"model": "another", "state": "x", "questions": self.request["questions"]})
        with self.assertRaises(BadRequest):
            service.infer({"state": "x", "questions": {}})

    def test_backend_identity_check(self):
        service = Service(self.config_path)
        calls, fake = self.backend(.95)
        with mock.patch("inference.server.get_json", side_effect=fake):
            service.ensure_backends(wait_seconds=0)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
