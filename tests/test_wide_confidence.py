"""Wide confidence dataset adapters, exclusions and run statistics."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"benchmarks"))
import adapters  # noqa: E402
from run_confidence_wide import paired_ci  # noqa: E402

class WideDatasetTests(unittest.TestCase):
    def test_source_gold_is_complete_only(self):
        choice={"type":"choice","criteria":{"A":"a","B":"b"}}
        full=adapters.original_gold({"q":choice},{"q":{"probabilities":{"A":1.0}}})
        self.assertEqual(full["q"],{"A":1.0,"B":0.0})
        with self.assertRaises(adapters.SkipGold):
            adapters.original_gold({"q":choice},{"q":{"probabilities":{"A":0.95}}})
        with self.assertRaises(adapters.SkipGold):
            adapters.original_gold({"q":{"type":"score","criteria":["0","1"]}},
                                   {"q":{"mean":0.5,"variance":0.2}})
        binary=adapters.original_gold({"q":{"type":"noul"}},{"q":{"noul":0.2}})
        self.assertAlmostEqual(binary["q"]["false"],0.8)

    def test_helmo_adapter_and_explicit_exclusions(self):
        q={"q":{"type":"noul","instructions":"Answer"}}
        records=[
            {"state":"x","topic":"science","questions":json.dumps(q),
             "gold":json.dumps({"q":{"noul":1}})},
            {"state":"y","topic":"science",
             "questions":json.dumps({"q":{"type":"score","criteria":["0","1"]}}),
             "gold":json.dumps({"q":{"mean":0.4,"variance":0.2}})}
        ]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"source.jsonl"
            path.write_text("".join(json.dumps(r)+"\n" for r in records))
            rows,digest,audit=adapters.normalize(path,"helmo_gold_v1",with_audit=True)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]["gold"]["q"]["true"],1)
            self.assertEqual(audit["excluded_score_moments_not_categorical"],1)
            self.assertEqual(audit["source_rows"],2)
            self.assertEqual(len(digest),64)

    def test_hellaswag(self):
        row={"ind":4,"__source_row_number":5,"ctx":"A man walks.",
             "endings":["A","B","C","D"],"label":"2","activity_label":"walking"}
        r=adapters.hellaswag_gold(row)
        self.assertEqual(r["gold"]["continuation"],"2")
        self.assertEqual(len(r["request"]["questions"]["continuation"]["criteria"]),4)

    def test_paired_ci_clusters_repeats(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with (root/"measurements.jsonl").open("w") as stream:
                for ident in ("a","b"):
                    for rep in (0,1):
                        for arm,ms,quality in (("strong_only",100,0.9),
                                                ("fast_only",10,0.6),
                                                ("cascade",70,0.8)):
                            stream.write(json.dumps({"id":ident,"repeat":rep,
                                "arm":arm,"latency_ms":ms,
                                "quality":{"question_expected_accuracy":quality,
                                           "questions":1}})+"\n")
            ci=paired_ci(root,draws=60)
            self.assertAlmostEqual(ci["speedup_pct_cascade_vs_ultra"],30)
            self.assertAlmostEqual(ci["delta_quality_pp_cascade_vs_ultra"],-10)
            self.assertAlmostEqual(ci["delta_quality_pp_cascade_vs_pico"],20)

if __name__=="__main__":
    unittest.main()
