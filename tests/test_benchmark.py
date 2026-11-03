"""CPU-only acceptance tests of runner, validation, routing, Gold, reproducibility."""
import json
import threading
from http.server import ThreadingHTTPServer
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
            "schema":"cascadeone-cascade-v0.1","id":"mock-cascade",
            "routing":{"type":"confidence_threshold","parameters":{"threshold":0.7}},
            "fast":{"url":"http://fast","model_id":"test/fast","revision":"sha","backend":{"implementation":"test-fast","mode":"mock"}},
            "strong":{"url":"http://strong","model_id":"test/strong","revision":"sha","backend":{"implementation":"test-strong","mode":"mock"}}
        }))
        req={"state":"Test","questions":{"q":{"type":"choice",
                "criteria":{"a":"A","b":"B"}}}}
        self.dataset.write_text("".join(json.dumps({"id":str(i),"request":req,
                                             "gold":{"q":"a"},"domain":"medical" if i%2 else "legal"})+"\n" for i in range(3)))
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
    def serve_inference(self, strong_alone=False):
        from inference.server import Service, make_handler
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(
            Service(self.config, api_token="test-secret-benchmark-token-987",
                    benchmark_mode=True, strong_alone=strong_alone)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return "http://127.0.0.1:" + str(server.server_port)

    def test_run_and_verify(self):
        url=self.serve_inference()
        with mock.patch("inference.server.get_json",side_effect=self.fake_request),\
             mock.patch.object(runner,"Telemetry",SimulatedTelemetry):
            result=runner.run(self.config,self.dataset,self.root/"r1",repeats=2,warmup=0,
                              inference_url=url,api_token="test-secret-benchmark-token-987")
            assert result["arms"]["cascade"]["escalation_rate"]==1
            assert result["by_domain"]["legal"]["cascade"]["n"]==4
            assert result["by_domain"]["medical"]["cascade"]["n"]==2
            self.assertEqual(json.loads((self.root/"r1"/"manifest.json").read_text())["routing"]["type"],"confidence_threshold")
            self.assertTrue(all("routing" in json.loads(line) for line in (self.root/"r1"/"measurements.jsonl").read_text().splitlines()))
            assert result["arms"]["cascade"]["question_expected_accuracy"]==1
            assert result["arms"]["strong_only"]["n"]==6
            assert result["arms"]["fast_only"]["n"]==6
            self.assertEqual(result["arms"]["fast_only"]["escalation_rate"],0)
            self.assertEqual(json.loads((self.root/"r1"/"manifest.json").read_text())["models"]["fast"]["backend"]["implementation"],"test-fast")
            self.assertEqual(result["arms"]["fast_only"]["question_expected_accuracy"],1)
            self.assertIn("engine_mean_ms",result["arms"]["cascade"])
            self.assertIn("proxy_overhead_mean_ms",result["arms"]["cascade"])
            self.assertEqual(json.loads((self.root/"r1"/"manifest.json").read_text())["execution"],
                             "systemone_inference_proxy_http")
            measurements=[json.loads(x) for x in (self.root/"r1"/"measurements.jsonl").read_text().splitlines()]
            self.assertTrue(all("engine_ms" in x and "proxy_overhead_ms" in x for x in measurements))
            assert set(result["by_domain"])=={"medical","legal"}
            assert result["by_domain"]["medical"]["cascade"]["n"]==2
            assert validate_result.validate(self.root/"r1")["valid"]
            with self.assertRaises(FileExistsError):
                runner.run(self.config,self.dataset,self.root/"r1",warmup=0,
                           inference_url=url,api_token="test-secret-benchmark-token-987")
            alone_url=self.serve_inference(strong_alone=True)
            with mock.patch("inference.server.get_json",side_effect=self.fake_request):
                alone=runner.run(self.config,self.dataset,self.root/"r0",repeats=2,warmup=0,
                                 phase="strong-alone",inference_url=alone_url,
                                 api_token="test-secret-benchmark-token-987")
            assert alone["phase"]=="strong-alone"
            comparison=compare_runs.compare(self.root/"r0",self.root/"r1")
            assert comparison["gold_delta_pp"]==0
            with (self.root/"r1"/"measurements.jsonl").open("a") as f: f.write("{}\n")
            with self.assertRaisesRegex(ValueError,"SHA mismatch"):
                validate_result.validate(self.root/"r1")
    def test_runner_is_independent_of_confidence(self):
        config=json.loads(self.config.read_text())
        config["routing"]={"type":"always_fast"}
        self.config.write_text(json.dumps(config))
        def no_conf(url,request=None,timeout=360):
            answer=self.fake_request(url,request,timeout)
            if url=="http://fast/v1/systemone":
                answer["answers"]["q"].pop("confidence",None)
            return answer
        url=self.serve_inference()
        with mock.patch("inference.server.get_json",side_effect=no_conf),\
             mock.patch.object(runner,"Telemetry",SimulatedTelemetry):
            report=runner.run(self.config,self.dataset,self.root/"confidence-free",
                              repeats=1,warmup=0,inference_url=url,
                              api_token="test-secret-benchmark-token-987")
        self.assertEqual(report["arms"]["cascade"]["escalation_rate"],0)
        self.assertTrue(validate_result.validate(self.root/"confidence-free")["valid"])
        rows=[json.loads(s) for s in (self.root/"confidence-free"/"measurements.jsonl").read_text().splitlines()]
        self.assertEqual({r["routing"]["policy"] for r in rows if r["arm"]=="cascade"},{"always_fast"})
        self.assertTrue(all(r["routing"]["signals"]=={} for r in rows if r["arm"]=="cascade"))

    def test_setup_resolves_registered_source_and_adapter(self):
        import io
        from contextlib import redirect_stdout
        output=self.root/"setup-output"
        setup=ROOT/"benchmarks/smoke-general__laya-clef__confidence-min-t0p900000/benchmark.json"
        with mock.patch.object(sys,"argv",["runner.py","--setup",str(setup),"--output",str(output)]),\
             mock.patch.object(runner,"run",return_value={"ok":True}) as call,\
             redirect_stdout(io.StringIO()):
            runner.main()
        args,kwargs=call.call_args
        self.assertEqual(Path(args[0]).name,"cascade.yaml")
        self.assertEqual(Path(args[1]).name,"source.jsonl")
        self.assertEqual(kwargs["adapter"],"typed_decisions_v1")
        self.assertEqual(kwargs["selection"],{"type":"all"})
        self.assertEqual(kwargs["setup_identity"]["id"],"smoke-general__laya-clef__confidence-min-t0p900000")
        self.assertEqual(kwargs["dataset_identity"]["mode"],"bundled")

    def test_gold_and_probability_guards(self):
        req=runner.load_dataset(self.dataset)[0][0]["request"]
        pred=self.fake_request("http://fast/v1/systemone")
        self.assertTrue(runner.validate_result(req,pred,fast=True))
        self.assertEqual(runner.score_gold(req,pred,{"q":{"a":0.3,"b":0.7}})["question_expected_accuracy"],0.3)
        pred["answers"]["q"]["probabilities"]["a"]=1.9
        with self.assertRaises(ValueError):
            runner.validate_result(req,pred,fast=True)

if __name__=="__main__":
    unittest.main()
