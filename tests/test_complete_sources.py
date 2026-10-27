"""Verify original raw source groups without depending on downloaded data."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"datasets"))
from download_source import manifest  # noqa: E402
from download_external import identities  # noqa: E402
from catalog import get  # noqa: E402


class CompleteOriginalSources(unittest.TestCase):
    def test_complete_group_manifests(self):
        expected={"banking77":2,"clinc150":9,"better-jev-bench":11}
        self.assertEqual(len(identities()),sum(expected.values()))
        for source,count in expected.items():
            with self.subTest(source=source):
                info=manifest(source)
                self.assertEqual(len(info["files"]),count)
                self.assertEqual(info["preservation"],
                                 "complete_original_source_splits_no_filtering_no_deduplication")
                self.assertTrue((ROOT/"datasets/sources"/source/"download.py").is_file())

    def test_all_original_bank_and_clinc_splits(self):
        bank={get(item["id"])["source"]["file"] for item in manifest("banking77")["files"]}
        self.assertEqual(bank,{"data/train-00000-of-00001.parquet",
                               "data/test-00000-of-00001.parquet"})
        clinc={get(item["id"])["source"]["file"] for item in manifest("clinc150")["files"]}
        self.assertEqual(clinc,{f"{config}/{part}-00000-of-00001.parquet"
                  for config in ("plus","small","imbalanced")
                  for part in ("train","validation","test")})

    def test_all_original_bjb_heldout_tasks(self):
        names={Path(item["path"]).name for item in manifest("better-jev-bench")["files"]}
        self.assertEqual(names,{x+".jsonl.gz" for x in (
            "atari_head","banking77","cfpb_complaints","civil_comments",
            "clinc150","cuad","go_emotions","ledgar","massive","os_atlas",
            "screenspot_v2")})
        import json
        setups=[p for p in (ROOT/"benchmarks").glob("*/benchmark.json")
                if json.loads(p.read_text()).get("suite")=="external"
                and p.parent.name.startswith("better-jev-bench-")]
        self.assertEqual(len(setups),8)
        self.assertFalse(any("screenspot" in s.parent.name for s in setups))


if __name__=="__main__":
    unittest.main()
