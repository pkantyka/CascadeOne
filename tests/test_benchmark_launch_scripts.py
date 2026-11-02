"""CPU-only tests of the GPU benchmark launch scripts and inference readiness."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import wait_models


@unittest.skipUnless(shutil.which("sh"), "POSIX sh is required for shell launch tests")
class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        fake = self.work / "docker"
        fake.write_text(
            '#!/bin/sh\n'
            'printf "%s\\n" "$*" >> "$CASCADEONE_TEST_DOCKER_LOG"\n'
            'printf "%s\\n" "$CASCADEONE_BENCHMARK_TOKEN" >> "$CASCADEONE_TEST_TOKEN_LOG"\n'
            'case "$*" in *" --verify"*) if [ "${CASCADEONE_TEST_FAIL_VERIFY:-0}" = 1 ]; then exit 9; fi ;; esac\n',
            encoding="utf-8",
        )
        fake.chmod(0o755)
        self.log = self.work / "docker.log"
        self.tokens = self.work / "tokens.log"
        self.env = dict(os.environ,
                        PATH=str(self.work) + os.pathsep + os.environ["PATH"],
                        CASCADEONE_TEST_DOCKER_LOG=str(self.log),
                        CASCADEONE_TEST_TOKEN_LOG=str(self.tokens),
                        CASCADEONE_BENCHMARK_TOKEN="test-token-for-shell-integration-1234")

    def launch(self, script, env=None, output=None):
        return subprocess.run([
            "sh", str(ROOT / "benchmarks" / script),
            str(output or self.work / "new-output")],
            cwd=ROOT, env=env or self.env, capture_output=True, text=True, timeout=30)

    def test_both_suites_preflight_before_model_start(self):
        for script, suite, source in (
            ("run_confidence_wide.sh", "wide", "download_wide.py"),
            ("run_confidence_external.sh", "external", "download_external.py"),
        ):
            with self.subTest(suite=suite):
                self.log.unlink(missing_ok=True)
                self.tokens.unlink(missing_ok=True)
                result = self.launch(script)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = self.log.read_text().splitlines()
                self.assertEqual(len(calls), 7)
                self.assertIn("build benchmark", calls[0])
                self.assertIn("--no-deps benchmark python datasets/" + source, calls[1])
                self.assertIn("--verify", calls[2])
                self.assertIn("--suite " + suite + " --dry-run", calls[3])
                self.assertIn("up -d --build --wait --wait-timeout 660 fast strong inference", calls[4])
                self.assertIn("benchmarks/wait_models.py --inference-url http://inference:18358", calls[5])
                self.assertIn("--suite " + suite + " --output-root", calls[6])
                self.assertIn("--no-deps benchmark", calls[6])
                self.assertTrue(all("--profile inference" in c for c in calls))
                self.assertTrue(all(t == self.env["CASCADEONE_BENCHMARK_TOKEN"]
                                    for t in self.tokens.read_text().splitlines()))

    def test_verify_failure_never_starts_gpu_models(self):
        env = dict(self.env, CASCADEONE_TEST_FAIL_VERIFY="1")
        result = self.launch("run_confidence_wide.sh", env)
        self.assertNotEqual(result.returncode, 0)
        calls = self.log.read_text().splitlines()
        self.assertEqual(len(calls), 3)
        self.assertFalse(any(" up " in c for c in calls))

    def test_existing_output_is_rejected_before_docker(self):
        existing = self.work / "already"
        existing.mkdir()
        result = self.launch("run_confidence_external.sh", output=existing)
        self.assertEqual(result.returncode, 2)
        self.assertIn("already exists", result.stderr)
        self.assertFalse(self.log.exists())


    def test_generates_token_when_not_supplied(self):
        openssl = self.work / "openssl"
        openssl.write_text('#!/bin/sh\nprintf "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\\n"\n')
        openssl.chmod(0o755)
        env = dict(self.env)
        env.pop("CASCADEONE_BENCHMARK_TOKEN")
        result = self.launch("run_confidence_wide.sh", env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(t == "0123456789abcdef" * 4
                            for t in self.tokens.read_text().splitlines()))

    def test_rejects_short_token_before_docker(self):
        result = self.launch("run_confidence_external.sh",
                             dict(self.env, CASCADEONE_BENCHMARK_TOKEN="short"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("at least 16", result.stderr)
        self.assertFalse(self.log.exists())

class ReadinessTests(unittest.TestCase):
    def test_checks_models_and_exact_inference_profile(self):
        config = json.loads(wait_models.CONFIG.read_text())
        sha = wait_models.sha(wait_models.CONFIG)
        calls = []

        def fake(url, timeout=15):
            calls.append(url)
            if url.endswith("/health"):
                return {"status": "ok"}
            if url.startswith("http://inference"):
                return {"models": [{"name": config["id"], "cascade_config_sha256": sha,
                                    "residency": "colocated",
                                    "backend_attestations": {"fast": {}, "strong": {}}}]}
            role = "fast" if "fast" in url else "strong"
            return {"models": [{"name": config[role]["model_id"].split("/")[-1],
                                "mode": config[role]["mode"], "orders": config[role]["orders"]}]}

        with mock.patch.object(wait_models, "request_json", side_effect=fake), \
             mock.patch.object(wait_models, "check_identity",
                               side_effect=lambda url, cfg: {"verified": True}):
            wait_models.check_ready("http://inference:18358")
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[-1], "http://inference:18358/v1/models")

    def test_inference_mismatch_blocks_benchmark(self):
        config = json.loads(wait_models.CONFIG.read_text())
        def fake(url, timeout=15):
            if url.endswith("/health"): return {"status": "ok"}
            return {"models": [{"name": config["id"],
                                "cascade_config_sha256": "bad",
                                "residency": "colocated",
                                "backend_attestations": {"fast": {}, "strong": {}}}]}
        with mock.patch.object(wait_models, "request_json", side_effect=fake), \
             mock.patch.object(wait_models, "check_identity", return_value={}):
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                wait_models.check_ready("http://inference:18358")


if __name__ == "__main__":
    unittest.main()


class ComposeQuickstartTests(unittest.TestCase):
    def test_laya_cuda_image_has_nonempty_default(self):
        compose = (ROOT / "cascades/laya-clef/compose.yaml").read_text()
        dockerfile = (ROOT / "cascades/laya-clef/Dockerfile").read_text()
        self.assertIn("${CASCADEONE_CUDA_BASE:-nvcr.io/nvidia/vllm:26.07-py3}", compose)
        self.assertIn("ARG CUDA_BASE=nvcr.io/nvidia/vllm:26.07-py3", dockerfile)
        self.assertNotIn("CASCADEONE_CUDA_BASE:?", compose)

    def test_both_inference_services_have_healthchecks(self):
        for cascade in ("laya-clef", "basal-pico-ultra"):
            compose = (ROOT / "cascades" / cascade / "compose.yaml").read_text()
            with self.subTest(cascade=cascade):
                self.assertIn("    healthcheck:", compose)
                self.assertIn("/health", compose)
