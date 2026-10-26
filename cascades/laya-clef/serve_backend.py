"""Individual pinned Laya or Clef HTTP service, independently measured."""
import argparse
from fastapi import FastAPI
import torch
import uvicorn
from cascadeone.core.types import DecisionRequest
from cascadeone.backends.laya import LayaFastBackend
from cascadeone.backends.clef import ClefStrongBackend

def make_app(which):
    backend=LayaFastBackend() if which=="fast" else ClefStrongBackend()
    app=FastAPI(title="CascadeOne single-backend service")
    @app.get("/health")
    def health():
        return {"status":"ok"}
    @app.get("/v1/models")
    def models():
        return {"models":[{
            "name":backend.identity.model_id.split("/")[-1],
            "revision":backend.identity.revision,
            "mode":"reference","orders":1
        }]}
    @app.post("/v1/systemone")
    def infer(body:DecisionRequest):
        with torch.inference_mode():
            if which=="fast":
                if any(q.type=="choice" and len(q.option_ids())>10 for q in body.questions.values()):
                    return {"fast_eligible":False,"reason":"calibrated_option_count_exceeded"}
                prepared=backend.prepare_for_routing(body)
                if not prepared.eligible:
                    return {"fast_eligible":False,"reason":prepared.reason}
                result=backend.infer(body,prepared=prepared.prepared)
            else:
                result=backend.infer(body)
        answers={}
        for qid,probabilities in result.probabilities.items():
            answers[qid]={"probabilities":probabilities,
                          "confidence":result.answer_confidence.get(qid)}
        return {"fast_eligible":True,"answers":answers}
    return app

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--which",choices=("fast","strong"),required=True)
    args=parser.parse_args()
    uvicorn.run(make_app(args.which),host="0.0.0.0",port=18357)
