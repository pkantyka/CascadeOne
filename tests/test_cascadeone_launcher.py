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

    def test_smoke_stops_models_after_validation_by_default(self):
        result = self.call("benchmark", "smoke", "--cascade", "laya-clef/confidence",
                           "--output", "output/cleanup-smoke")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertIn("tools/validate_result.py", calls[-2])
        self.assertEqual(calls[-1], "compose -f cascades/laya-clef/confidence/compose.yaml "
                         "--profile inference --profile benchmark down")

    def test_evaluate_stops_models_after_validation_by_default(self):
        result = self.call("benchmark", "evaluate", "--cascade", "basal-pico-ultra/confidence",
                           "--output", "output/cleanup-evaluate")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertIn("tools/validate_result.py", calls[-2])
        self.assertEqual(calls[-1], "compose -f cascades/basal-pico-ultra/confidence/compose.yaml "
                         "--profile inference --profile benchmark down")

    def test_keep_running_preserves_stack_for_smoke_and_evaluate(self):
        for action in ("smoke", "evaluate"):
            with self.subTest(action=action):
                self.commands.unlink(missing_ok=True)
                result = self.call("benchmark", action, "--cascade", "laya-clef/confidence",
                                   "--output", "output/keep-" + action, "--keep-running")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(any(" up -d --build --wait" in x for x in self.calls()))
                self.assertTrue(any("tools/validate_result.py" in x for x in self.calls()))
                self.assertFalse(any(x.endswith(" down") for x in self.calls()))
                self.assertIn("Keeping model services running", result.stderr)

    def test_stop_only_touches_selected_compose_without_token(self):
        env = dict(self.env, CASCADEONE_API_TOKEN="")
        result = self.call("benchmark", "stop", "--cascade", "laya-clef/pre-router", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [
            "compose version",
            "compose -f cascades/laya-clef/pre-router/compose.yaml "
            "--profile inference --profile benchmark down"
        ])
        self.assertFalse((self.work / ".cascadeone").exists())

    def test_stop_rejects_benchmark_options_before_docker(self):
        for options in (("--keep-running",), ("--repeats", "3"), ("--cuda-base", "any"),
                        ("--benchmark", "demo")):
            with self.subTest(options=options):
                self.commands.unlink(missing_ok=True)
                result = self.call("benchmark", "stop", "--cascade",
                                   "laya-clef/confidence", *options)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.commands.exists(), result.stderr)
        for mode, action in (("inference", "start"), ("benchmark", "list")):
            with self.subTest(mode=mode, action=action):
                self.commands.unlink(missing_ok=True)
                result = self.call(mode, action, "--cascade",
                                   "laya-clef/confidence", "--keep-running")
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.commands.exists(), result.stderr)

    def test_cleanup_on_failed_validation_preserves_failure_status(self):
        fake = self.work / "docker"
        fake.write_text(fake.read_text() +
                        '\ncase "$*" in *"tools/validate_result.py"*) exit 47 ;; esac\n')
        result = self.call("benchmark", "smoke", "--cascade", "laya-clef/confidence",
                           "--output", "output/fail-validation")
        self.assertEqual(result.returncode, 47, result.stderr)
        self.assertIn("tools/validate_result.py", self.calls()[-2])
        self.assertTrue(self.calls()[-1].endswith("--profile inference --profile benchmark down"))

    def test_interrupt_during_benchmark_stops_models(self):
        fake = self.work / "docker"
        fake.write_text(fake.read_text() +
                        '\ncase "$*" in *"tools/validate_result.py"*) kill -TERM "$PPID" ;; esac\n')
        result = self.call("benchmark", "smoke", "--cascade", "laya-clef/confidence",
                           "--output", "output/interrupt-test")
        self.assertEqual(result.returncode, 143, result.stderr)
        self.assertTrue(self.calls()[-1].endswith("--profile inference --profile benchmark down"))

    def test_cleanup_on_failed_compose_up(self):
        fake = self.work / "docker"
        fake.write_text(fake.read_text() +
                        '\ncase "$*" in *" --wait-timeout 660"*) exit 29 ;; esac\n')
        result = self.call("benchmark", "smoke", "--cascade", "laya-clef/confidence",
                           "--output", "output/fail-start")
        self.assertEqual(result.returncode, 29, result.stderr)
        self.assertTrue(any(" up -d --build --wait" in x for x in self.calls()))
        self.assertTrue(self.calls()[-1].endswith("--profile inference --profile benchmark down"))

    def test_failed_cleanup_is_reported_and_fails_successful_benchmark(self):
        fake = self.work / "docker"
        fake.write_text(fake.read_text() +
                        '\ncase "$*" in *" --profile inference --profile benchmark down"*) exit 31 ;; esac\n')
        result = self.call("benchmark", "smoke", "--cascade", "laya-clef/confidence",
                           "--output", "output/fail-shutdown")
        self.assertEqual(result.returncode, 31, result.stderr)
        self.assertIn("WARNING: Compose shutdown failed", result.stderr)
        self.assertIn("./cascadeone benchmark stop --cascade laya-clef/confidence",
                      result.stderr)

    def test_suite_is_not_required_for_named_benchmark(self):
        result = self.call("benchmark", "evaluate", "--cascade", "basal-pico-ultra/confidence",
                           "--benchmark", BASAL_CIVIL, "--output", "output/test-named-unique")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = "\n".join(self.calls())
        self.assertIn("datasets/catalog.py acquire bjb-civil-comments-heldout-v1", commands)
        self.assertIn("--setup benchmarks/" + BASAL_CIVIL + "/benchmark.json", commands)
        self.assertIn("tools/validate_result.py output/test-named-unique", commands)
        self.assertNotIn("--suite", commands)
        self.assertTrue(all(x == "testing-only-token-0123456789" for x in self.tokens.read_text().splitlines()))

    def test_suite_is_optional_and_selects_multiple_setups(self):
        result = self.call("benchmark", "evaluate", "--cascade", "basal-pico-ultra/confidence",
                           "--suite", "external", "--output", "output/test-suite-unique")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = "\n".join(self.calls())
        self.assertIn("datasets/download_external.py --verify", commands)
        self.assertIn("run_confidence_wide.py --suite external --cascade basal-pico-ultra/confidence --dry-run", commands)
        self.assertIn("run_confidence_wide.py --suite external --cascade basal-pico-ultra/confidence --output-root", commands)
        self.assertNotIn("tools/validate_result.py output/test-suite-unique", commands)

    def test_default_single_evaluation(self):
        result = self.call("benchmark", "evaluate", "--cascade", "laya-clef/confidence",
                           "--output", "output/test-laya-eval")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = "\n".join(self.calls())
        self.assertIn("datasets/catalog.py acquire bjb-civil-comments-heldout-v1", commands)
        self.assertIn("--cascade cascades/laya-clef/confidence/cascade.yaml --dataset-id bjb-civil-comments-heldout-v1", commands)
        self.assertNotIn("--suite", commands)

    def test_model_independent_dataset_and_adapter(self):
        result = self.call("benchmark", "evaluate", "--cascade", "laya-clef/confidence",
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
        result = self.call("inference", "start", "--cascade", "basal-pico-ultra/confidence",
                           "--cuda-base", image)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(i == image for i in self.images.read_text().splitlines()[1:]))
        self.assertTrue(all(t == self.env["CASCADEONE_API_TOKEN"] for t in self.tokens.read_text().splitlines()[1:]))
        self.assertIn("--wait-timeout 660", "\n".join(self.calls()))
        self.assertNotIn("--wait-timeout 660 fast strong inference", "\n".join(self.calls()))

    def test_old_cuda_environment_is_ignored_unless_requested(self):
        result = self.call("inference", "start", "--cascade", "laya-clef/confidence")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(i == "" for i in self.images.read_text().splitlines()[1:]))

    def test_smoke_has_no_suite_and_is_validated(self):
        result = self.call("benchmark", "smoke", "--cascade", "laya-clef/confidence",
                           "--output", "output/test-laya-smoke")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("smoke-general__laya-clef__confidence-min-t0p900000/benchmark.json", "\n".join(self.calls()))
        self.assertIn("tools/validate_result.py", "\n".join(self.calls()))

    def test_list_with_or_without_suite_needs_no_docker(self):
        all_items = self.call("benchmark", "list", "--cascade", "basal-pico-ultra/confidence")
        self.assertEqual(all_items.returncode, 0, all_items.stderr)
        self.assertIn("banking77-test__", all_items.stdout)
        selected = self.call("benchmark", "list", "--cascade", "basal-pico-ultra/confidence",
                             "--suite", "external")
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertIn(BASAL_CIVIL, selected.stdout)
        self.assertNotIn("smoke-general__", selected.stdout)
        self.assertFalse(self.commands.exists())

    def test_benchmark_list_has_headers_and_aligned_columns(self):
        result = self.call("benchmark", "list", "--cascade", "basal-pico-ultra/confidence")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertGreater(len(lines), 2)
        columns = [line.split("\t") for line in lines]
        self.assertTrue(all(len(row) == 3 for row in columns))
        self.assertEqual([cell.strip() for cell in columns[0]],
                         ["BENCHMARK", "DATASET", "SUITE"])
        self.assertEqual(len({len(row[0]) for row in columns}), 1)
        self.assertEqual(len({len(row[1]) for row in columns}), 1)
        self.assertEqual({row[2] for row in columns[1:]},
                         {"smoke", "source", "wide", "external"})

        selected = self.call("benchmark", "list", "--cascade",
                             "basal-pico-ultra/confidence", "--suite", "external")
        self.assertEqual(selected.returncode, 0, selected.stderr)
        filtered = [line.split("\t") for line in selected.stdout.splitlines()]
        self.assertEqual([c.strip() for c in filtered[0]],
                         ["BENCHMARK", "DATASET", "SUITE"])
        self.assertEqual(len(filtered), 11)
        self.assertTrue(all(row[2] == "external" for row in filtered[1:]))
        self.assertFalse(self.commands.exists())

    def test_cascade_list_formats_yaml_profiles_without_docker(self):
        result = self.call("cascade", "list")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split("\t") for line in result.stdout.splitlines()]
        self.assertEqual([cell.strip() for cell in rows[0]],
                         ["CASCADE", "FAST MODEL", "STRONG MODEL", "ROUTING"])
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(len(row) == 4 for row in rows))
        self.assertEqual(len({len(row[0]) for row in rows}), 1)
        self.assertEqual(len({len(row[1]) for row in rows}), 1)
        self.assertEqual(len({len(row[2]) for row in rows}), 1)
        values = {row[0].strip(): row for row in rows[1:]}
        self.assertEqual(set(values), {"laya-clef/confidence", "laya-clef/pre-router",
                                      "basal-pico-ultra/confidence"})
        self.assertIn("Remek/basal-omni-pico", values["basal-pico-ultra/confidence"][1])
        self.assertIn("Cloudflare/clef-flash", values["laya-clef/confidence"][2])
        self.assertEqual(values["laya-clef/confidence"][3], "confidence_threshold")
        self.assertEqual(values["laya-clef/pre-router"][3], "max_questions_pre_router")
        self.assertFalse(self.commands.exists())

    def test_cascade_list_rejects_selection_options(self):
        for option in (("--cascade", "laya-clef/confidence"), ("--suite", "external"),
                       ("--cuda-base", "example"), ("--repeats", "3")):
            with self.subTest(option=option):
                result = self.call("cascade", "list", *option)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.commands.exists())

    def test_pre_router_variant_smoke_uses_matching_compose_and_setup(self):
        result = self.call("benchmark", "smoke", "--cascade", "laya-clef/pre-router",
                           "--output", "output/test-pre-router")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = "\n".join(self.calls())
        self.assertIn("-f cascades/laya-clef/pre-router/compose.yaml", calls)
        self.assertIn("--setup benchmarks/smoke-general__laya-clef__pre-router-max-questions-1/benchmark.json", calls)
        self.assertIn("tools/validate_result.py output/test-pre-router", calls)

    def test_variant_owns_inference_stack(self):
        result = self.call("inference", "start", "--cascade", "laya-clef/pre-router")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-f cascades/laya-clef/pre-router/compose.yaml --profile inference up",
                      "\n".join(self.calls()))

    def test_invalid_cascade_identifiers_are_rejected_before_docker(self):
        for invalid in ("laya-clef", "basal-pico-ultra",
                        "laya-clef/../../other", "/confidence",
                        "laya-clef/", "laya-clef/unknown",
                        "no-such-pair/confidence"):
            with self.subTest(value=invalid):
                result = self.call("inference", "start", "--cascade", invalid)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.commands.exists())

    def test_named_benchmark_from_other_variant_fails_before_docker(self):
        result = self.call("benchmark", "evaluate",
                           "--cascade", "laya-clef/pre-router",
                           "--benchmark", "smoke-general__laya-clef__confidence-min-t0p900000")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.commands.exists())

    def test_new_variant_requires_no_launcher_change(self):
        (self.work / "cascades").unlink()
        custom = self.work / "cascades" / "other-pair" / "unusual-router"
        custom.mkdir(parents=True)
        shutil.copy2(ROOT / "cascades/config.py", self.work / "cascades/config.py")
        shutil.copy2(ROOT / "cascades/laya-clef/confidence/cascade.yaml",
                     custom / "cascade.yaml")
        (custom / "compose.yaml").write_text("name: example\nservices: {}\n")
        listing = self.call("cascade", "list")
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertIn("other-pair/unusual-router", listing.stdout)
        start = self.call("inference", "start",
                          "--cascade", "other-pair/unusual-router")
        self.assertEqual(start.returncode, 0, start.stderr)
        self.assertIn("-f cascades/other-pair/unusual-router/compose.yaml",
                      "\n".join(self.calls()))

    def test_errors_before_starting_models(self):
        cases = (
            ("benchmark", "evaluate", "--cascade", "basal-pico-ultra/confidence",
             "--dataset-id", "banking77-test-v1"),
            ("benchmark", "evaluate", "--cascade", "basal-pico-ultra/confidence",
             "--dataset-id", "banking77-test-v1", "--adapter", "banking77_test_gold_v1", "--suite", "wide"),
            ("benchmark", "evaluate", "--cascade", "laya-clef/confidence", "--suite", "external"),
            ("benchmark", "evaluate", "--cascade", "laya-clef/confidence", "--benchmark", BASAL_CIVIL),
            ("benchmark", "smoke", "--cascade", "laya-clef/confidence", "--suite", "wide"),
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
        self.assertEqual(len(benchmark_setups("external", "basal-pico-ultra/confidence")), 10)
        self.assertEqual(len(benchmark_setups("external", "laya-clef/confidence")), 0)
        self.assertEqual(len(benchmark_setups("smoke", "laya-clef/confidence")), 1)

    def test_env_token_never_creates_or_changes_token_file(self):
        token_path = self.work / ".cascadeone" / "token"
        result = self.call("inference", "start", "--cascade", "laya-clef/confidence")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(token_path.exists())
        token_path.parent.mkdir()
        token_path.write_text("preexisting-local-token-123456\n")
        supplied = "externally-supplied-token-987654"
        env = dict(self.env, CASCADEONE_API_TOKEN=supplied)
        result = self.call("inference", "start", "--cascade", "laya-clef/confidence", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(token_path.read_text(), "preexisting-local-token-123456\n")
        self.assertEqual(self.tokens.read_text().splitlines()[-1], supplied)

    def test_file_token_generated_reused_and_private(self):
        env = dict(self.env)
        env.pop("CASCADEONE_API_TOKEN")
        token_path = self.work / ".cascadeone" / "token"
        first = self.call("inference", "start", "--cascade", "laya-clef/confidence", env=env)
        self.assertEqual(first.returncode, 0, first.stderr)
        token = token_path.read_text().strip()
        self.assertGreaterEqual(len(token), 16)
        self.assertEqual(token_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(token_path.parent.stat().st_mode & 0o777, 0o700)
        second = self.call("benchmark", "smoke", "--cascade", "laya-clef/confidence",
                           "--output", "output/test-persisted-token", env=env)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(token_path.read_text().strip(), token)
        self.assertTrue(all(t == token for t in self.tokens.read_text().splitlines() if t))

    def test_explicit_empty_token_is_not_replaced_or_saved(self):
        env = dict(self.env, CASCADEONE_API_TOKEN="")
        result = self.call("inference", "start", "--cascade", "laya-clef/confidence", env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("at least 16", result.stderr)
        self.assertFalse((self.work / ".cascadeone").exists())
        self.assertFalse(any(" up " in call for call in self.calls()))

    def test_benchmark_endpoints_enabled_only_for_benchmark_launcher(self):
        result = self.call("inference", "start", "--cascade", "laya-clef/confidence")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.modes.read_text().splitlines()[-1], "0")
        self.modes.unlink()
        result = self.call("benchmark", "smoke", "--cascade", "laya-clef/confidence",
                           "--output", "output/test-mode-gate")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.modes.read_text().splitlines()[-1], "1")

    def test_token_generation_is_internal(self):
        env = dict(self.env)
        env.pop("CASCADEONE_API_TOKEN")
        result = self.call("benchmark", "smoke", "--cascade", "laya-clef/confidence",
                           "--output", "output/test-autotoken", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        tokens = self.tokens.read_text().splitlines()
        self.assertGreaterEqual(len(tokens), 3)
        self.assertTrue(all(t == tokens[1] and len(t) >= 16 for t in tokens[1:]))

if __name__ == "__main__":
    unittest.main()
