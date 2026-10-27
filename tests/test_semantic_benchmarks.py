"""Contract tests: one named benchmark always contains all three arms."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"benchmarks"))
sys.path.insert(0,str(ROOT/"datasets"))
import identity  # noqa: E402
import run_confidence_wide as suite  # noqa: E402
import download_wide  # noqa: E402


class SemanticBenchmarkTests(unittest.TestCase):
    def test_all_benchmark_names_and_strategies_match_configs(self):
        expected={"external":10,"wide":5,"smoke":4}
        directories=list((ROOT/"benchmarks").glob("*/benchmark.json"))
        self.assertEqual(len(directories),19)
        self.assertEqual(len({p.parent.name for p in directories}),19)
        for group,count in expected.items():
            self.assertEqual(len(suite.benchmark_setups(group)),count)
        for path in directories:
            with self.subTest(name=path.parent.name):
                setup=json.loads(path.read_text(encoding="utf-8"))
                cascade=json.loads((ROOT/setup["cascade"]).read_text(encoding="utf-8"))
                self.assertEqual(path.parent.name,setup["id"])
                self.assertEqual(path.parent.name,identity.benchmark_id(setup,cascade))
                self.assertEqual(path.parent.name.count("__"),2)
                self.assertNotEqual(cascade["fast"]["model_id"],cascade["strong"]["model_id"])
                self.assertTrue((path.parent/"run.py").is_file())
                self.assertTrue((path.parent/"README.md").is_file())

    def test_wide_downloader_discovers_all_six_pinned_files(self):
        # Five selections, but procedural data uses two shards.
        self.assertEqual(len(download_wide.SETUPS),5)
        self.assertEqual(len(download_wide.dataset_ids()),5)
        for p in download_wide.SETUPS:
            self.assertEqual(json.loads(p.read_text())["suite"],"wide")

    def test_single_benchmark_wrapper_preflight_without_models(self):
        name="smoke-general__laya-clef__confidence-min-t0p900000"
        wrapper=ROOT/"benchmarks"/name/"run.py"
        proc=subprocess.run([sys.executable,str(wrapper),"--dry-run"],cwd=ROOT,
                            capture_output=True,text=True,check=True)
        self.assertIn(name,proc.stdout)
        self.assertIn('"rows": 3',proc.stdout)

    def test_suite_forces_all_three_arms(self):
        name="smoke-general__laya-clef__confidence-min-t0p900000"
        required={"fast_only":{},"strong_only":{},"cascade":{}}
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"reports"
            with (mock.patch.object(suite,"run",return_value={"arms":required}) as invoked,
                  mock.patch.object(suite,"validate",return_value={"valid":True}),
                  mock.patch.object(suite,"paired_ci",return_value={"quality":0}),
                  mock.patch.object(sys,"argv",["run_confidence_wide.py","--suite","smoke",
                       "--setup-id",name,"--output-root",str(target),"--repeats","1"]),
                  redirect_stdout(io.StringIO())):
                suite.main()
            self.assertEqual(invoked.call_args.kwargs["phase"],"colocated")
            data=json.loads((target/"suite.json").read_text())
            self.assertEqual(set(data["required_arms"]),set(required))
            self.assertEqual(data["runs"][0]["setup"],name)
            self.assertEqual(set(data["runs"][0]["arms"]),set(required))

    def test_reject_incomplete_model_comparison(self):
        name="smoke-general__laya-clef__confidence-min-t0p900000"
        incomplete={"strong_only":{},"cascade":{}}
        with tempfile.TemporaryDirectory() as tmp:
            with (mock.patch.object(suite,"run",return_value={"arms":incomplete}),
                  mock.patch.object(suite,"validate",return_value={"valid":True}),
                  mock.patch.object(sys,"argv",["run_confidence_wide.py","--suite","smoke",
                       "--setup-id",name,"--output-root",str(Path(tmp)/"reports")]),
                  redirect_stdout(io.StringIO())):
                with self.assertRaisesRegex(ValueError,"all three arms"):
                    suite.main()


if __name__=="__main__":
    unittest.main()
