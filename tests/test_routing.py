"""Confidence and confidence-free policies do not change benchmark measurement."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"cascades"))
from runtime import Cascade
from policies import build_policy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.req={"state":"test","questions":{"q":{"type":"noul"}}}
        self.config={"fast":{"url":"fast"},"strong":{"url":"strong"}}

    def execute(self,routing,req=None,confidence=None,fast_eligible=True):
        config=dict(self.config,routing=routing)
        calls=[]
        def invoke(url,request,timeout,fast):
            calls.append(url)
            if fast:
                if not fast_eligible:
                    return {"fast_eligible":False,"reason":"unsupported_input"},1.0
                answer={"probabilities":{"true":0.9,"false":0.1}}
                if confidence is not None: answer["confidence"]=confidence
                return {"answers":{"q":answer}},1.0
            return {"answers":{"q":{"probabilities":{"true":0.8,"false":0.2}}}},5.0
        return Cascade(config).execute(req or self.req,30,invoke),calls

    def test_confidence_threshold(self):
        cfg={"type":"confidence_threshold","parameters":{"threshold":0.7}}
        result,calls=self.execute(cfg,confidence=0.9)
        self.assertFalse(result.escalated)
        self.assertEqual(calls,["fast"])
        result,calls=self.execute(cfg,confidence=0.2)
        self.assertTrue(result.escalated)
        self.assertEqual(calls,["fast","strong"])
        result,calls=self.execute(cfg,confidence=None)
        self.assertTrue(result.escalated)
        self.assertEqual(result.route["reason"],"missing_or_invalid_confidence")

    def test_confidence_free_fast(self):
        result,calls=self.execute({"type":"always_fast"},confidence=None)
        self.assertFalse(result.escalated)
        self.assertEqual(calls,["fast"])
        self.assertEqual(result.route["signals"],{})

    def test_confidence_free_prerouting(self):
        policy={"type":"max_questions_pre_router","parameters":{"max_questions":1}}
        request={"state":"x","questions":{"a":{"type":"noul"},"b":{"type":"noul"}}}
        result,calls=self.execute(policy,req=request)
        self.assertTrue(result.escalated)
        self.assertEqual(calls,["strong"])
        self.assertFalse(result.route["fast_executed"])
        result,calls=self.execute(policy,confidence=None)
        self.assertFalse(result.escalated)
        self.assertEqual(calls,["fast"])

    def test_always_strong_and_eligibility(self):
        result,calls=self.execute({"type":"always_strong"})
        self.assertTrue(result.escalated)
        self.assertEqual(calls,["strong"])
        result,calls=self.execute({"type":"always_fast"},fast_eligible=False)
        self.assertTrue(result.escalated)
        self.assertEqual(calls,["fast","strong"])

    def test_invalid_router_contract_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"unknown routing policy"):
            build_policy({"type":"magic"})
        with self.assertRaisesRegex(ValueError,"obsolete"):
            Cascade(dict(self.config,threshold=0.5,routing={"type":"always_fast"}))


if __name__=="__main__":
    unittest.main()
