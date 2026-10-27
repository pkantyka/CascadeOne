"""Offline, deterministic tests of public benchmark source contracts."""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"datasets"))
sys.path.insert(0,str(ROOT/"benchmarks"))
import catalog  # noqa: E402
import adapters  # noqa: E402
import download_external  # noqa: E402


class ExternalTests(unittest.TestCase):
    def test_complete_raw_sources_and_independent_setups(self):
        ids=download_external.identities()
        self.assertEqual(len(ids),22)
        for source_id in ids:
            entry=catalog.get(source_id)
            self.assertEqual(entry["mode"],"downloadable")
            self.assertEqual(len(entry["source"]["revision"]),40)
            self.assertEqual(len(entry["source"]["sha256"]),64)
            self.assertTrue(catalog.source_url(entry["source"]).startswith("https://"))
        setups=([ROOT/"benchmarks/banking77/setup.json",
                 ROOT/"benchmarks/clinc150/setup.json"]+
                sorted((ROOT/"benchmarks/better-jev-bench/setups").glob("*.json")))
        self.assertEqual(len(setups),10)
        for setup_path in setups:
            setup=json.loads(setup_path.read_text())
            self.assertIn(setup["dataset"]["id"],ids)
            self.assertIn(setup["adapter"],adapters.ADAPTERS)
            self.assertEqual(setup["selection"],{"type":"all"})

    def test_github_revision_path_guard(self):
        s={"provider":"github_raw","repo":"asp616848/better-jev-bench",
           "revision":"a"*40,"file":"bench/heldout/cuad.jsonl.gz"}
        self.assertIn("/"+"a"*40+"/",catalog.source_url(s))
        for revision,file in [("main",s["file"]),("a"*40,"../private.jsonl")]:
            with self.subTest(revision=revision,file=file):
                with self.assertRaises(ValueError):
                    catalog.source_url({**s,"revision":revision,"file":file})

    def test_original_classlabel_is_not_shortened(self):
        names=[f"category_{i}" for i in range(77)]
        row={"text":"Where is my card?","label":11,"__source_row_number":0,
             "__class_names":{"label":names}}
        record=adapters.banking77_test_gold(row)
        self.assertEqual(len(record["request"]["questions"]["intent"]["criteria"]),77)
        self.assertEqual(record["gold"]["intent"],"11")
        with self.assertRaises(ValueError):
            adapters.banking77_test_gold({**row,"__class_names":{"label":names[:-1]}})
        names=[f"intent_{i}" for i in range(151)]
        row={"text":"hello","intent":150,"__source_row_number":0,
             "__class_names":{"intent":names}}
        record=adapters.clinc150_plus_test_gold(row)
        self.assertEqual(len(record["request"]["questions"]["intent"]["criteria"]),151)
        self.assertEqual(record["gold"]["intent"],"150")

    def test_bjb_mapping_is_explicit(self):
        bjb={"item_id":"abc","dataset":"cuad",
             "request":{"state":"This Agreement assigns rights.",
                        "questions":{"test":{"type":"noul",
                          "instructions":"Check clause",
                          "options":["Yes","No"]}}},
             "expected":{"test":"Yes"}}
        row=adapters.better_jev_heldout_gold(bjb)
        self.assertEqual(row["gold"],{"test":"true"})
        self.assertEqual(row["request"]["questions"]["test"]["criteria"],
                         {"true":"Yes","false":"No"})
        with self.assertRaises(ValueError):
            adapters.better_jev_heldout_gold({**bjb,"expected":{"different":"Yes"}})
        bjb["request"]["questions"]["test"]={"type":"choice","options":["A","B"],
                                               "instructions":"Choose"}
        bjb["expected"]["test"]="B"
        row=adapters.better_jev_heldout_gold(bjb)
        self.assertEqual(row["gold"],{"test":"1"})


if __name__=="__main__":
    unittest.main()
