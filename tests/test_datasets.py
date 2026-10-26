"""Dataset identity stays source-only; benchmark adapters own processing."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"datasets"))
sys.path.insert(0,str(ROOT/"benchmarks"))
import catalog
import adapters


class SourceRegistryTests(unittest.TestCase):
    def make_source(self):
        rows=[]
        for i in range(50):
            q={"q":{"type":"score","criteria":["low","high"]},
               "b":{"type":"noul"},
               "c":{"type":"choice","criteria":{"left":"left","right":"right"}}}
            rows.append({"state_id":f"state-{i}",
                         "domain":"medical" if i%2 else "legal",
                         "state":json.dumps({"record":i}),
                         "questions":json.dumps(q),
                         "gold":json.dumps({"q":i%2,"b":bool(i%2),"c":"left" if i%2 else "right"})})
        return "".join(json.dumps(x)+"\n" for x in rows).encode("utf-8")

    def test_bundled_raw_source_unchanged(self):
        source,identity=catalog.resolve("smoke-general-v1")
        self.assertEqual(identity["mode"],"bundled")
        self.assertEqual(source.name,"source.jsonl")
        rows,digest=adapters.normalize(source,"typed_decisions_v1",{"type":"all"})
        self.assertEqual(len(rows),3)
        self.assertEqual(len(digest),64)

    def test_bundled_hash_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            f=root/"source.jsonl"
            f.write_text("a\n")
            manifest=root/"registry.json"
            manifest.write_text(json.dumps({"schema":catalog.SCHEMA,"datasets":[{
                "id":"test-bundle","mode":"bundled",
                "source":{"path":"source.jsonl","sha256":catalog.sha(f)}
            }]}))
            catalog.resolve("test-bundle",manifest)
            f.write_text("b\n")
            with self.assertRaisesRegex(ValueError,"SHA mismatch"):
                catalog.resolve("test-bundle",manifest)

    def test_pinned_download_keeps_exact_raw_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            raw=self.make_source()
            digest=hashlib.sha256(raw).hexdigest()
            reg=root/"registry.json"
            reg.write_text(json.dumps({"schema":catalog.SCHEMA,"datasets":[{
                "id":"hf-test","mode":"downloadable",
                "source":{"provider":"huggingface","repo":"someone/example",
                          "revision":"b"*40,"file":"data/train.jsonl","sha256":digest}
            }]}))
            cache=root/"cache"
            calls=[]
            def transport(request,timeout):
                calls.append(request.full_url)
                return io.BytesIO(raw)
            path,identity=catalog.resolve("hf-test",reg,cache,download=True,opener=transport)
            self.assertEqual(path.read_bytes(),raw)
            self.assertEqual(identity["source_sha256"],digest)
            self.assertEqual(len(calls),1)
            self.assertEqual(catalog.resolve("hf-test",reg,cache)[0],path)
            normalized,prepared_hash=adapters.normalize(path,"n4ze3m_gold_v1")
            self.assertEqual(len(normalized),50)
            self.assertEqual(len(prepared_hash),64)
            self.assertEqual(normalized[0]["gold"]["b"],"false")
            self.assertEqual(normalized[1]["gold"]["b"],"true")
            selection={"type":"hash_range","start":0.0,"end":0.5,"seed":20261026}
            filtered,selection_hash=adapters.normalize(path,"n4ze3m_gold_v1",selection)
            self.assertTrue(0<len(filtered)<50)
            self.assertEqual(selection_hash,adapters.normalize(path,"n4ze3m_gold_v1",selection)[1])
            self.assertEqual(path.read_bytes(),raw)
            with path.open("ab") as stream: stream.write(b"x")
            with self.assertRaisesRegex(ValueError,"SHA mismatch"):
                catalog.resolve("hf-test",reg,cache)

    def test_download_failure_leaves_no_raw_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            reg=root/"registry.json"
            reg.write_text(json.dumps({"schema":catalog.SCHEMA,"datasets":[{
                "id":"bad","mode":"downloadable",
                "source":{"provider":"huggingface","repo":"someone/example",
                          "revision":"b"*40,"file":"data/raw.jsonl","sha256":"0"*64}
            }]}))
            cache=root/"cache"
            with self.assertRaisesRegex(ValueError,"source SHA mismatch"):
                catalog.resolve("bad",reg,cache,download=True,
                                opener=lambda request,timeout:io.BytesIO(b"untrusted"))
            self.assertFalse((cache/"raw/bad/data/raw.jsonl").exists())

    def test_benchmark_adapter_gold_and_path_guards(self):
        sample={"state_id":"a","state":"test",
                "questions":'{"q":{"type":"noul"}}',
                "gold":'{"different":true}'}
        with self.assertRaisesRegex(ValueError,"Gold mismatch"):
            adapters.n4ze3m_gold(sample)
        for source in ("../escape","/etc/passwd","..\\escape"):
            with self.assertRaises(ValueError):
                catalog.safe_relative(source)
        with self.assertRaisesRegex(ValueError,"selection"):
            adapters.normalize(ROOT/"datasets/bundled/smoke/source.jsonl",
                               selection={"type":"something_unknown"})

if __name__=="__main__":
    unittest.main()
