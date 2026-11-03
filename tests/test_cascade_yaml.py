"""Cascade definition YAML migration and launch-profile contract tests."""
from pathlib import Path
import json
import sys
import yaml
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cascades"))
from config import load_cascade, list_cascades


class CascadeYamlTests(unittest.TestCase):
    def test_only_yaml_defines_cascades(self):
        names = {name for name, _ in list_cascades(ROOT)}
        self.assertEqual(names, {"basal-pico-ultra/confidence", "laya-clef/confidence", "laya-clef/pre-router"})
        for name in names:
            self.assertTrue((ROOT / "cascades" / name / "cascade.yaml").is_file())
            self.assertTrue((ROOT / "cascades" / name / "compose.yaml").is_file())
            self.assertFalse((ROOT / "cascades" / name / "benchmark.json").exists())
            config = load_cascade(ROOT / "cascades" / name / "cascade.yaml")
            self.assertNotEqual(config["fast"]["model_id"], config["strong"]["model_id"])
            self.assertIn("routing", config)

    def test_additional_pre_router_is_yaml_and_referenced(self):
        path = ROOT / "cascades/laya-clef/pre-router/cascade.yaml"
        variant = load_cascade(path)
        self.assertEqual(variant["routing"]["type"], "max_questions_pre_router")
        self.assertEqual(variant["routing"]["parameters"]["max_questions"], 1)
        setup = json.loads((ROOT / "benchmarks/smoke-general__laya-clef__pre-router-max-questions-1/benchmark.json").read_text())
        self.assertEqual(setup["cascade"], "cascades/laya-clef/pre-router/cascade.yaml")

    def test_all_benchmark_json_configs_reference_yaml(self):
        setups = sorted((ROOT / "benchmarks").glob("*/benchmark.json"))
        self.assertEqual(len(setups), 19)
        for path in setups:
            with self.subTest(setup=path.parent.name):
                setup = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(setup["cascade"].endswith(".yaml"))
                profile = load_cascade(ROOT / setup["cascade"])
                self.assertEqual(profile["schema"], "cascadeone-cascade-v0.1")

    def test_each_variant_has_independent_compose_and_correct_config(self):
        projects = set()
        for name, _ in list_cascades(ROOT):
            with self.subTest(variant=name):
                variant = ROOT / "cascades" / name
                compose = yaml.safe_load((variant / "compose.yaml").read_text())
                self.assertEqual(compose["name"],
                                 "cascadeone-community-" + name.replace("/", "-"))
                projects.add(compose["name"])
                services = compose["services"]
                for role in ("fast", "strong", "inference", "benchmark"):
                    self.assertIn(role, services)
                for role in ("inference", "benchmark"):
                    self.assertEqual(
                        (variant / services[role]["build"]["context"]).resolve(), ROOT)
                    self.assertTrue(any(v.split(":")[0] == "../../.."
                                        for v in services[role]["volumes"]))
                command = services["inference"]["command"]
                self.assertEqual(command[command.index("--cascade") + 1],
                                 f"cascades/{name}/cascade.yaml")
                setup_args = services["benchmark"]["command"]
                setup = ROOT / setup_args[setup_args.index("--setup") + 1]
                self.assertEqual(json.loads(setup.read_text())["cascade"],
                                 f"cascades/{name}/cascade.yaml")
        self.assertEqual(len(projects), 3)
        for pair in ("laya-clef", "basal-pico-ultra"):
            self.assertFalse((ROOT / "cascades" / pair / "compose.yaml").exists())
            self.assertFalse((ROOT / "cascades" / pair / "cascade.yaml").exists())

    def test_rejects_invalid_or_unsafe_cascade_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cascade.yaml"
            for invalid in ("schema: unknown\n", "[]\n", "!!python/object/apply:os.system ['echo unsafe']\n"):
                with self.subTest(value=invalid):
                    path.write_text(invalid)
                    with self.assertRaises((ValueError, yaml.YAMLError)):
                        load_cascade(path)

    def test_inference_and_benchmark_containers_install_yaml_parser(self):
        inference = (ROOT / "inference/Dockerfile").read_text()
        runner = (ROOT / "benchmarks/Dockerfile").read_text()
        self.assertIn("PyYAML==6.0.2", inference)
        self.assertIn("PyYAML==6.0.2", runner)
        for name in ("basal-pico-ultra/confidence", "laya-clef/confidence", "laya-clef/pre-router"):
            compose = (ROOT / "cascades" / name / "compose.yaml").read_text()
            self.assertIn("dockerfile: inference/Dockerfile", compose)
            self.assertIn(f"cascades/{name}/cascade.yaml", compose)
            self.assertIn("context: ../../..", compose)
            self.assertIn("../../..:/workspace", compose)


if __name__ == "__main__":
    unittest.main()
