"""CPU-only acceptance tests of runner, validation, routing, Gold, reproducibility."""
import json
import urllib.error
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"benchmarks"))
sys.path.insert(0,str(ROOT/"tools"))
import runner
import validate_result
import compare_runs

class SimulatedTelemetry:
    def __init__(self,path):
        self.path=Path(path)
        self.summary={"samples":1,"scope":"host-wide_not_attributable_to_single_model",
                      "cpu_util_percent":None,"host_ram_used_mib":None,
                      "vram_used_mib":None,"gpu_util_percent":None,"gpu_power_watts":None}
        self.gpu_description=None
    def __enter__(self):
        self.path.write_text('{"utc":"mock","gpus":[]}\n')
        return self
    def __exit__(self,*args): return False

class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.config=self.root/"config.json"
        self.dataset=self.root/"input.jsonl"
        self.config.write_text(json.dumps({
            "schema":"cascadeone-cascade-v0.1","id":"mock-cascade","threshold":0.7,
            "fast":{"url":"http://fast","model_id":"test/fast","revision":"sha"},
            "strong":{"url":"http://strong","model_id":"test/strong","revision":"sha"}
        }))
        req={"state":"Test","questions":{"q":{"type":"choice",
                "criteria":{"a":"A","b":"B"}}}}
        self.dataset.write_text("".join(json.dumps({"id":str(i),"request":req,
                                             "gold":{"q":"a"}})+"\n" for i in range(3)))
    def tearDown(self):
        self.tmp.cleanup()
    def fake_request(self,url,request=None,timeout=360):
        if url=="http://fast/health":
            raise urllib.error.URLError("not running")
        if url.endswith("/v1/models"):
            return {"models":[{"name":"fast" if "fast" in url else "strong"}]}
        if "fast" in url:
            return {"answers":{"q":{"probabilities":{"a":0.6,"b":0.4},"confidence":0.6}}}
        return {"answers":{"q":{"probabilities":{"a":0.9,"b":0.1},"confidence":0.9}}}
    def test_run_and_verify(self):
        with mock.patch.object(runner,"request_json",side_effect=self.fake_request),\
             mock.patch.object(runner,"Telemetry",SimulatedTelemetry):
            result=runner.run(self.config,self.dataset,self.root/"r1",repeats=2,warmup=0)
            assert result["arms"]["cascade"]["escalation_rate"]==1
            assert result["arms"]["cascade"]["question_expected_accuracy"]==1
            assert result["arms"]["strong_only"]["n"]==6
            assert validate_result.validate(self.root/"r1")["valid"]
            with self.assertRaises(FileExistsError):
                runner.run(self.config,self.dataset,self.root/"r1",warmup=0)
            with mock.patch.object(runner,"request_json",side_effect=self.fake_request):
                alone=runner.run(self.config,self.dataset,self.root/"r0",repeats=2,warmup=0,phase="strong-alone")
            assert alone["phase"]=="strong-alone"
            comparison=compare_runs.compare(self.root/"r0",self.root/"r1")
            assert comparison["gold_delta_pp"]==0
            with (self.root/"r1"/"measurements.jsonl").open("a") as f: f.write("{}\n")
            with self.assertRaisesRegex(ValueError,"SHA mismatch"):
                validate_result.validate(self.root/"r1")
    def test_gold_and_probability_guards(self):
        req=runner.load_dataset(self.dataset)[0]["request"]
        pred=self.fake_request("http://fast/v1/systemone")
        self.assertAlmostEqual(runner.validate_result(req,pred,fast=True),0.6)
        self.assertEqual(runner.score_gold(req,pred,{"q":{"a":0.3,"b":0.7}})["question_expected_accuracy"],0.3)
        pred["answers"]["q"]["probabilities"]["a"]=1.9
        with self.assertRaises(ValueError):
            runner.validate_result(req,pred,fast=True)

if __name__=="__main__":
    unittest.main()
