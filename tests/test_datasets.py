"""Two delivery modes, offline HF transport, immutable datasets and SHA checks."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"datasets"))
import catalog

class DatasetTests(unittest.TestCase):
    def test_bundled_has_frozen_calibration_test(self):
        info=catalog.prepare("smoke-general-v1")
        self.assertEqual(info["mode"],"bundled")
        self.assertEqual(set(info["splits"]),{"calibration","test"})
        self.assertEqual(catalog.validate_rows(info["splits"]["test"])["rows"],3)
        p,identity=catalog.resolved("smoke-general-v1")
        self.assertEqual(p,info["splits"]["test"])
        self.assertEqual(identity["mode"],"bundled")
        self.assertEqual(len(identity["manifest_sha256"]),64)

    def test_bundled_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            fixture=base/"test.jsonl"
            fixture.write_text((ROOT/"datasets/bundled/smoke/test.jsonl").read_text())
            registry=base/"registry.json"
            registry.write_text(json.dumps({"schema":catalog.SCHEMA,"datasets":[{
                "id":"test-bundle","mode":"bundled",
                "splits":{"test":{"path":"test.jsonl","sha256":catalog.sha(fixture),"rows":3}}
            }]}))
            with mock.patch.object(catalog,"ROOT",base):
                self.assertEqual(catalog.verify("test-bundle",registry)["mode"],"bundled")
                fixture.write_text("{}\n")
                with self.assertRaisesRegex(ValueError,"checksum"):
                    catalog.verify("test-bundle",registry)

    def make_source(self):
        rows=[]
        for i in range(50):
            question={
                "q":{"type":"score","instructions":"Rate","criteria":["low","high"]},
                "b":{"type":"noul","instructions":"Is it present?"},
                "c":{"type":"choice","criteria":{"left":"left","right":"right"}}
            }
            rows.append({
                "state_id":f"state-{i}","domain":"medical" if i%2 else "legal",
                "state":json.dumps({"record":i}),"questions":json.dumps(question),
                "gold":json.dumps({"q":i%2,"b":bool(i%2),"c":"left" if i%2 else "right"})
            })
        return "".join(json.dumps(row)+"\n" for row in rows).encode("utf-8")

    def test_download_prepare_and_verify_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            raw=self.make_source()
            digest=hashlib.sha256(raw).hexdigest()
            entry={
                "id":"fixture-hf","mode":"downloadable","domain":"mixed",
                "adapter":catalog.ADAPTER,"split_seed":20261026,
                "source":{"provider":"huggingface","repo":"example/fixture",
                          "revision":"a"*40,"file":"data/train.jsonl","sha256":digest}
            }
            reg=root/"registry.json"
            reg.write_text(json.dumps({"schema":catalog.SCHEMA,"datasets":[entry]}))
            cache=root/"cache"
            calls=[]
            def fake_download(req,timeout):
                calls.append(req.full_url)
                return io.BytesIO(raw)
            output=catalog.prepare("fixture-hf",reg,cache,opener=fake_download)
            self.assertEqual(len(calls),1)
            self.assertIn("example/fixture/resolve/",calls[0])
            self.assertEqual(set(output["splits"]),{"calibration","test"})
            identity=catalog.verify("fixture-hf",reg,cache)
            self.assertEqual(identity["splits"],output["splits"])
            self.assertEqual(catalog.prepare("fixture-hf",reg,cache,opener=fake_download),output)
            self.assertEqual(len(calls),1)
            c=catalog.validate_rows(output["splits"]["calibration"])
            t=catalog.validate_rows(output["splits"]["test"])
            self.assertEqual(c["gold_requests"],c["rows"])
            self.assertEqual(t["gold_requests"],t["rows"])
            self.assertFalse(c["ids"]&t["ids"])
            self.assertIn("medical",set(c["domains"])|set(t["domains"]))
            with output["splits"]["test"].open("a",encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError,"SHA mismatch"):
                catalog.verify("fixture-hf",reg,cache)

    def test_download_checksum_failure_never_publishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            entry={
                "id":"fixture-bad","mode":"downloadable","adapter":catalog.ADAPTER,
                "split_seed":20261026,
                "source":{"provider":"huggingface","repo":"example/fixture",
                          "revision":"a"*40,"file":"data/train.jsonl","sha256":"0"*64}
            }
            reg=root/"registry.json"
            reg.write_text(json.dumps({"schema":catalog.SCHEMA,"datasets":[entry]}))
            cache=root/"cache"
            with self.assertRaisesRegex(ValueError,"raw file SHA mismatch"):
                catalog.prepare("fixture-bad",reg,cache,
                                opener=lambda req,timeout:io.BytesIO(self.make_source()))
            self.assertFalse((cache/"prepared/fixture-bad").exists())
            self.assertFalse((cache/"raw/fixture-bad/data/train.jsonl").exists())

    def test_bad_path_and_mismatched_gold(self):
        for path in ("../secret","/tmp/a","..\\a"):
            with self.assertRaises(ValueError):
                catalog.safe_relative(path)
        wrong={"state_id":"x","state":"A",
               "questions":'{"q":{"type":"noul"}}',"gold":'{"wrong":true}'}
        with self.assertRaisesRegex(ValueError,"request/Gold"):
            catalog.convert_n4ze3m(wrong)

if __name__=="__main__":
    unittest.main()
