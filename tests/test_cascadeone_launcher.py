"""CPU-only tests of the user-facing CascadeOne launcher with a fake Docker."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cascadeone"
BASAL_CIVIL = "better-jev-bench-civil-comments-heldout__basal-pico-ultra__confidence-min-t0p656352"

@unittest.skipUnless(shutil.which("sh") and shutil.which("python3"), "requires POSIX shell and Python3")
class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.launcher = self.work / "cascadeone"
        shutil.copy2(LAUNCHER, self.launcher)
        for name in ("cascades", "benchmarks"):
            (self.work / name).symlink_to(ROOT / name, target_is_directory=True)
        fake = self.work / "docker"
        fake.write_text(
            '#!/bin/sh\n'
            'printf "%s\\n" "$*" >> "$CASCADEONE_TEST_COMMAND_LOG"\n'
            'printf "%s\\n" "${CASCADEONE_API_TOKEN-}" >> "$CASCADEONE_TEST_TOKEN_LOG"\n'
            'printf "%s\\n" "${CASCADEONE_CUDA_BASE-}" >> "$CASCADEONE_TEST_BASE_LOG"\n'
            'printf "%s\\n" "${CASCADEONE_BENCHMARK_MODE-}" >> "$CASCADEONE_TEST_MODE_LOG"\n',
            encoding="utf-8"
        )
        fake.chmod(0o755)
        self.commands = self.work / "commands"
        self.tokens = self.work / "tokens"
        self.images = self.work / "images"
        self.modes = self.work / "modes"
        self.env = dict(os.environ,
                        PATH=str(self.work) + os.pathsep + os.environ["PATH"],
                        CASCADEONE_TEST_COMMAND_LOG=str(self.commands),
                        CASCADEONE_TEST_TOKEN_LOG=str(self.tokens),
                        CASCADEONE_TEST_BASE_LOG=str(self.images),
                        CASCADEONE_TEST_MODE_LOG=str(self.modes),
                        CASCADEONE_API_TOKEN="testing-only-token-0123456789",
                        CASCADEONE_CUDA_BASE="unused-invalid-base")

    def call(self, *args, env=None):
        return subprocess.run(["sh", str(self.launcher), *args], cwd=self.work,
                              env=env or self.env, capture_output=True,
                              text=True, timeout=20)

    def calls(self):
        return self.commands.read_text().splitlines() if self.commands.exists() else []

    def test_suite_is_not_required_for_named_benchmark(self):
        result = self.call("benchmark", "evaluate", "--cascade", "basal-pico-ultra",
                           "--benchmark", BASAL_CIVIL, "--output", "output/test-named-unique")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = "\n".join(self.calls())
        self.assertIn("datasets/catalog.py acquire bjb-civil-comments-heldout-v1", commands)
        self.assertIn("--setup benchmarks/" + BASAL_CIVIL + "/benchmark.json", commands)
        self.assertIn("tools/validate_result.py output/test-named-unique", commands)
        self.assertNotIn("--suite", commands)
        self.assertTrue(all(x == "testing-only-token-0123456789" for x in self.tokens.read_text().splitlines()))

    def test_suite_is_optional_and_selects_multiple_setups(self):
        result = self.call("benchmark", "evaluate", "--cascade", "basal-pico-ultra",
                           "--suite", "external", "--output", "output/test-suite-unique")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = "\n".join(self.calls())
        self.assertIn("datasets/download_external.py --verify", commands)
        self.assertIn("run_confidence_wide.py --suite external --cascade basal-pico-ultra --dry-run", commands)
        self.assertIn("run_confidence_wide.py --suite external --cascade basal-pico-ultra --output-root", commands)
        self.assertNotIn("tools/validate_result.py output/test-suite-unique", commands)

    def test_default_single_evaluation(self):
        result = self.call("benchmark", "evaluate", "--cascade", "laya-clef",
                           "--output", "output/test-laya-eval")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = "\n".join(self.calls())
        self.assertIn("datasets/catalog.py acquire bjb-civil-comments-heldout-v1", commands)
        self.assertIn("--cascade cascades/laya-clef/benchmark.json --dataset-id bjb-civil-comments-heldout-v1", commands)
        self.assertNotIn("--suite", commands)

    def test_model_independent_dataset_and_adapter(self):
        result = self.call("benchmark", "evaluate", "--cascade", "laya-clef",
                           "--dataset-id", "clinc150-plus-test-v1",
                           "--adapter", "clinc150_plus_test_gold_v1",
                           "--output", "output/test-custom")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = "\n".join(self.calls())
        self.assertIn("catalog.py acquire clinc150-plus-test-v1", commands)
        self.assertIn("--dataset-id clinc150-plus-test-v1 --adapter clinc150_plus_test_gold_v1", commands)
        self.assertNotIn("--suite", commands)

    def test_cuda_base_overrides_both_backends(self):
        image = "nvcr.io/nvidia/vllm:26.07-py3"
        result = self.call("inference", "start", "--cascade", "basal-pico-ultra",
                           "--cuda-base", image)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(i == image for i in self.images.read_text().splitlines()[1:]))
        self.assertTrue(all(t == self.env["CASCADEONE_API_TOKEN"] for t in self.tokens.read_text().splitlines()[1:]))
        self.assertIn("--wait-timeout 660 fast strong inference", "\n".join(self.calls()))

    def test_old_cuda_environment_is_ignored_unless_requested(self):
        result = self.call("inference", "start", "--cascade", "laya-clef")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(i == "" for i in self.images.read_text().splitlines()[1:]))

    def test_smoke_has_no_suite_and_is_validated(self):
        result = self.call("benchmark", "smoke", "--cascade", "laya-clef",
                           "--output", "output/test-laya-smoke")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("smoke-general__laya-clef__confidence-min-t0p900000/benchmark.json", "\n".join(self.calls()))
        self.assertIn("tools/validate_result.py", "\n".join(self.calls()))

    def test_list_with_or_without_suite_needs_no_docker(self):
        all_items = self.call("benchmark", "list", "--cascade", "basal-pico-ultra")
        self.assertEqual(all_items.returncode, 0, all_items.stderr)
        self.assertIn("banking77-test__", all_items.stdout)
        selected = self.call("benchmark", "list", "--cascade", "basal-pico-ultra",
                             "--suite", "external")
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertIn(BASAL_CIVIL, selected.stdout)
        self.assertNotIn("smoke-general__", selected.stdout)
        self.assertFalse(self.commands.exists())

    def test_errors_before_starting_models(self):
        cases = (
            ("benchmark", "evaluate", "--cascade", "basal-pico-ultra",
             "--dataset-id", "banking77-test-v1"),
            ("benchmark", "evaluate", "--cascade", "basal-pico-ultra",
             "--dataset-id", "banking77-test-v1", "--adapter", "banking77_test_gold_v1", "--suite", "wide"),
            ("benchmark", "evaluate", "--cascade", "laya-clef", "--suite", "external"),
            ("benchmark", "evaluate", "--cascade", "laya-clef", "--benchmark", BASAL_CIVIL),
            ("benchmark", "smoke", "--cascade", "laya-clef", "--suite", "wide"),
        )
        for case in cases:
            with self.subTest(case=case):
                self.commands.unlink(missing_ok=True)
                result = self.call(*case)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.commands.exists(), result.stderr)

    def test_suite_runner_filters_by_cascade(self):
        import sys
        sys.path.insert(0, str(ROOT / "benchmarks"))
        from run_confidence_wide import benchmark_setups
        self.assertEqual(len(benchmark_setups("external", "basal-pico-ultra")), 10)
        self.assertEqual(len(benchmark_setups("external", "laya-clef")), 0)
        self.assertEqual(len(benchmark_setups("smoke", "laya-clef")), 1)

    def test_env_token_never_creates_or_changes_token_file(self):
        token_path = self.work / ".cascadeone" / "token"
        result = self.call("inference", "start", "--cascade", "laya-clef")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(token_path.exists())
        token_path.parent.mkdir()
        token_path.write_text("preexisting-local-token-123456\n")
        supplied = "externally-supplied-token-987654"
        env = dict(self.env, CASCADEONE_API_TOKEN=supplied)
        result = self.call("inference", "start", "--cascade", "laya-clef", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(token_path.read_text(), "preexisting-local-token-123456\n")
        self.assertEqual(self.tokens.read_text().splitlines()[-1], supplied)

    def test_file_token_generated_reused_and_private(self):
        env = dict(self.env)
        env.pop("CASCADEONE_API_TOKEN")
        token_path = self.work / ".cascadeone" / "token"
        first = self.call("inference", "start", "--cascade", "laya-clef", env=env)
        self.assertEqual(first.returncode, 0, first.stderr)
        token = token_path.read_text().strip()
        self.assertGreaterEqual(len(token), 16)
        self.assertEqual(token_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(token_path.parent.stat().st_mode & 0o777, 0o700)
        second = self.call("benchmark", "smoke", "--cascade", "laya-clef",
                           "--output", "output/test-persisted-token", env=env)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(token_path.read_text().strip(), token)
        self.assertTrue(all(t == token for t in self.tokens.read_text().splitlines() if t))

    def test_explicit_empty_token_is_not_replaced_or_saved(self):
        env = dict(self.env, CASCADEONE_API_TOKEN="")
        result = self.call("inference", "start", "--cascade", "laya-clef", env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("at least 16", result.stderr)
        self.assertFalse((self.work / ".cascadeone").exists())
        self.assertFalse(any(" up " in call for call in self.calls()))

    def test_benchmark_endpoints_enabled_only_for_benchmark_launcher(self):
        result = self.call("inference", "start", "--cascade", "laya-clef")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.modes.read_text().splitlines()[-1], "0")
        self.modes.unlink()
        result = self.call("benchmark", "smoke", "--cascade", "laya-clef",
                           "--output", "output/test-mode-gate")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.modes.read_text().splitlines()[-1], "1")

    def test_token_generation_is_internal(self):
        env = dict(self.env)
        env.pop("CASCADEONE_API_TOKEN")
        result = self.call("benchmark", "smoke", "--cascade", "laya-clef",
                           "--output", "output/test-autotoken", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        tokens = self.tokens.read_text().splitlines()
        self.assertGreaterEqual(len(tokens), 3)
        self.assertTrue(all(t == tokens[1] and len(t) >= 16 for t in tokens[1:]))

if __name__ == "__main__":
    unittest.main()
