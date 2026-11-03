"""Measure three live decision HTTP arms: fast-only, strong-only and cascade.

Standard-library only; Python 3.12. Never uses cached predictions for timing.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import urllib.request
import urllib.error

from telemetry import Telemetry
from adapters import normalize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cascades'))
from config import load_cascade

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def utc():
    return datetime.now(timezone.utc).isoformat()

def percentile(values, fraction):
    a = sorted(values)
    if not a:
        raise ValueError("no measurements")
    pos = (len(a)-1)*fraction
    index = int(pos)
    return a[index] + (a[min(index+1,len(a)-1)]-a[index])*(pos-index)

def request_json(url, request=None, timeout=360):
    data = None if request is None else json.dumps(request,allow_nan=False).encode("utf-8")
    req = urllib.request.Request(url,data=data,headers={"Content-Type":"application/json"} if data else {})
    with urllib.request.urlopen(req,timeout=timeout) as res:
        return json.loads(res.read())

def check_identity(url, expected):
    info = request_json(url.rstrip("/")+"/v1/models",timeout=20)
    models = info.get("models",[])
    if len(models)!=1 or models[0].get("name")!=expected["model_id"].split("/")[-1]:
        raise ValueError("service identity mismatch at "+url)
    # Revision is pinned by service configuration but /v1/models may not attest weights.
    if models[0].get("mode") and expected.get("mode") and models[0]["mode"]!=expected["mode"]:
        raise ValueError("service mode mismatch")
    if models[0].get("orders") is not None and expected.get("orders") is not None and models[0]["orders"]!=expected["orders"]:
        raise ValueError("service order mismatch")
    return info

def option_ids(question):
    kind=question["type"]
    if kind=="choice": return list(question["criteria"])
    if kind=="noul": return ["false","true"]
    if kind=="score": return [str(i) for i in range(len(question["criteria"]))]
    raise ValueError("unsupported question type")

def validate_result(req,pred,fast=False):
    if fast and pred.get("fast_eligible") is False:
        return None
    answers=pred.get("answers")
    if not isinstance(answers,dict) or set(answers)!=set(req["questions"]):
        raise ValueError("missing or extra answers")
    for qid,question in req["questions"].items():
        answer=answers[qid]
        probs=answer.get("probabilities")
        if not isinstance(probs,dict) or set(probs)!=set(option_ids(question)):
            raise ValueError("prediction option set mismatch")
        if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 or x>1 for x in probs.values()):
            raise ValueError("invalid probabilities")
        if abs(sum(probs.values())-1)>0.002:
            raise ValueError("probabilities not normalized")
    return True

def score_gold(req,pred,gold):
    if gold is None:
        return None
    if set(gold)!=set(req["questions"]):
        raise ValueError("incomplete Gold labels")
    scores=[]
    hard=[]
    for qid,truth in gold.items():
        probs=pred["answers"][qid]["probabilities"]
        best=max(probs,key=probs.get)
        if isinstance(truth,str):
            if truth not in probs: raise ValueError("unknown Gold label")
            scores.append(float(best==truth))
            hard.append(best==truth)
        elif isinstance(truth,dict):
            if set(truth)!=set(probs) or any(type(v) not in (int,float) or not 0<=v<=1 for v in truth.values()) or abs(sum(truth.values())-1)>0.002:
                raise ValueError("invalid Gold distribution")
            scores.append(float(truth[best]))
        else:
            raise ValueError("Gold must be hard label or complete distribution")
    return {"question_expected_accuracy":statistics.mean(scores),
            "hard_request_exact":all(hard) if len(hard)==len(scores) else None,
            "questions":len(scores)}

def load_dataset(path,adapter="typed_decisions_v1",selection=None,with_audit=False):
    raw_rows,digest,audit=normalize(path,adapter,selection,with_audit=True)
    rows=[]
    seen=set()
    for item in raw_rows:
        key=item["id"]
        if not isinstance(key,str) or not key or key in seen: raise ValueError("invalid or duplicate request id")
        seen.add(key)
        request=item["request"]
        if not isinstance(request.get("questions"),dict) or not request["questions"]:
            raise ValueError("empty questions")
        for question in request["questions"].values():
            option_ids(question)
        if item.get("gold") is not None and set(item["gold"])!=set(request["questions"]):
            raise ValueError("Gold coverage mismatch")
        rows.append(item)
    if not rows: raise ValueError("dataset empty")
    return (rows,digest,audit) if with_audit else (rows,digest)

def timed(url,req,timeout,arm,token):
    """Measure client HTTP end-to-end plus engine-internal and backend HTTP timings."""
    if not token or len(token)<16:
        raise ValueError("CASCADEONE_API_TOKEN (16+ chars) is required")
    data=json.dumps(req,allow_nan=False).encode("utf-8")
    target=url.rstrip("/")+"/internal/benchmark/"+arm
    request=urllib.request.Request(target,data=data,headers={
        "Content-Type":"application/json","Authorization":"Bearer "+token})
    start=time.perf_counter_ns()
    with urllib.request.urlopen(request,timeout=timeout) as res:
        result=json.loads(res.read())
        headers=res.headers
    elapsed=(time.perf_counter_ns()-start)/1e6
    validate_result(req,result)
    if headers.get("X-CascadeOne-Arm") != arm:
        raise ValueError("proxy benchmark arm mismatch")
    route=json.loads(headers["X-CascadeOne-Route-Meta"])
    if arm!="cascade" and route.get("policy") != arm:
        raise ValueError("unexpected baseline policy")
    if arm!="cascade" and route["escalated"] is not (arm=="strong_only"):
        raise ValueError("unexpected baseline escalation")
    values=[float(headers[k]) for k in (
        "X-CascadeOne-Engine-Ms","X-CascadeOne-Fast-Ms","X-CascadeOne-Strong-Ms")]
    if any(not math.isfinite(v) or v<0 for v in values):
        raise ValueError("invalid engine timing")
    return result,elapsed,*values,route

def get_git():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL,timeout=3).strip()
    except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
        return None

def summarize(rows,phase):
    groups=defaultdict(list)
    for row in rows: groups[row["arm"]].append(row)
    result={}
    for arm,measure in groups.items():
        lat=[x["latency_ms"] for x in measure]
        internal=[x["engine_ms"] for x in measure]
        overhead=[x["proxy_overhead_ms"] for x in measure]
        quality=[x["quality"] for x in measure if x["quality"]]
        exact=[x["quality"]["hard_request_exact"] for x in measure if x["quality"] and x["quality"]["hard_request_exact"] is not None]
        result[arm]={
            "n":len(measure), "unique_requests":len(set(x["id"] for x in measure)),
            "mean_ms":statistics.mean(lat), "p50_ms":percentile(lat,0.5),
            "engine_mean_ms":statistics.mean(internal), "engine_p95_ms":percentile(internal,0.95),
            "proxy_overhead_mean_ms":statistics.mean(overhead),
            "proxy_overhead_p95_ms":percentile(overhead,0.95),
            "p95_ms":percentile(lat,0.95), "p99_ms":percentile(lat,0.99),
            "escalation_rate":sum(x["escalated"] for x in measure)/len(measure),
            "question_expected_accuracy":sum(x["question_expected_accuracy"]*x["questions"] for x in quality)/sum(x["questions"] for x in quality) if len(quality)==len(measure) else None,
            "hard_request_exact":statistics.mean(exact) if len(exact)==len(measure) else None,
            "sequential_requests_per_second":1000/statistics.mean(lat),
        }
    if phase=="colocated":
        baseline=result["strong_only"]
        for arm,metric in result.items():
            metric["mean_latency_reduction_vs_strong_only"] = 1-metric["mean_ms"]/baseline["mean_ms"]
            metric["engine_mean_latency_reduction_vs_strong_only"] = 1-metric["engine_mean_ms"]/baseline["engine_mean_ms"]
            metric["p95_ratio_vs_strong_only"]=metric["p95_ms"]/baseline["p95_ms"]
            if metric["question_expected_accuracy"] is not None and baseline["question_expected_accuracy"] is not None:
                metric["gold_delta_pp_vs_strong_only"]=100*(metric["question_expected_accuracy"]-baseline["question_expected_accuracy"])
            else:
                metric["gold_delta_pp_vs_strong_only"]=None
    return result

def run(config_path,dataset_path,output,repeats=2,warmup=3,timeout=360,phase="colocated",seed=20261026,dataset_identity=None,adapter="typed_decisions_v1",selection=None,setup_identity=None,inference_url=None,api_token=None):
    config=load_cascade(config_path)
    if config.get("schema")!="cascadeone-cascade-v0.1": raise ValueError("wrong cascade schema")
    if repeats<1 or warmup<0: raise ValueError("invalid repeats/warmup")
    inference_url=inference_url or os.environ.get("CASCADEONE_INFERENCE_URL","http://inference:18358")
    token=api_token or os.environ.get("CASCADEONE_API_TOKEN")
    if not token or len(token)<16:
        raise ValueError("CASCADEONE_API_TOKEN (16+ chars) is required to measure the shared inference engine")
    selection=selection or {"type":"all"}
    rows,normalized_sha,normalization_audit=load_dataset(dataset_path,adapter,selection,with_audit=True)
    dest=Path(output)
    if dest.exists(): raise FileExistsError("run output already exists")
    if phase not in ("colocated","strong-alone"): raise ValueError("invalid phase")
    identity=request_json(inference_url.rstrip("/")+"/v1/models",timeout=20)
    models=identity.get("models",[])
    if len(models)!=1 or models[0].get("name")!=config["id"] or models[0].get("cascade_config_sha256")!=sha(config_path):
        raise ValueError("inference service does not match the selected cascade")
    if models[0].get("residency","colocated")!=phase:
        raise ValueError("inference residency mode differs from benchmark phase")
    dest.mkdir(parents=True)
    samples=[]
    try:
        identities={"inference":identity}
        manifest={
            "schema":"cascadeone-run-v0.1","status":"complete",
            "created_utc":utc(),"cascade_id":config["id"],"phase":phase,
            "cascade_config_sha256":sha(config_path),"dataset_sha256":sha(dataset_path),
            "repeats":repeats,"warmup":warmup,"seed":seed,"routing":config["routing"],
            "models": {"fast":config["fast"],"strong":config["strong"]},
            "http_identity":identities,"code_commit":get_git(),
            "engine_version":"inference-systemone-v1",
            "inference_url":inference_url,
            "n_requests":len(rows),"execution":"systemone_inference_proxy_http",
            "dataset_identity":dataset_identity or {"mode":"local_unregistered","id":None,"split":None},
            "benchmark_setup":setup_identity or {"adapter":adapter,"selection":selection},
            "normalized_requests_sha256":normalized_sha,
            "normalization_audit":normalization_audit,
            "measurement_note":"latency_ms=client end-to-end HTTP including proxy; engine_ms=inside Cascade.execute including upstream HTTP; proxy_overhead_ms=difference; fast_ms/strong_ms include upstream HTTP; host telemetry is global",
            "gold_coverage":"all" if all(x.get("gold") is not None for x in rows) else "partial_or_none",
            "workload":{"domains":dict(Counter(x.get("domain","unspecified") for x in rows)),
                       "question_types":dict(Counter(q["type"] for x in rows for q in x["request"]["questions"].values())),
                       "mean_questions_per_request":sum(len(x["request"]["questions"]) for x in rows)/len(rows)}
        }
        (dest/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
        rng=random.Random(seed)
        arms=["strong_only"] if phase=="strong-alone" else ["strong_only","fast_only","cascade"]
        with Telemetry(dest/"telemetry.jsonl") as monitor:
            for arm in arms:
                for i in range(warmup):
                    timed(inference_url,rows[i%len(rows)]["request"],timeout,arm,token)
            with (dest/"measurements.jsonl").open("x",encoding="utf-8") as stream:
                for rep in range(repeats):
                    batch=rows[:]
                    rng.shuffle(batch)
                    for case in batch:
                        order=arms[:]
                        rng.shuffle(order)
                        for arm in order:
                            req=case["request"]
                            selected,elapsed,engine_ms,fast_ms,strong_ms,route=timed(
                                inference_url,req,timeout,arm,token)
                            escalated=route["escalated"]
                            if route.get("policy") != (config["routing"]["type"] if arm=="cascade" else arm):
                                raise ValueError("inference routing policy mismatch")
                            proxy_overhead_ms=elapsed-engine_ms
                            measure={
                                "id":case["id"],"repeat":rep,"arm":arm,"escalated":escalated,
                                "domain":case.get("domain","unspecified"),
                                "routing":route,"latency_ms":elapsed,
                                "engine_ms":engine_ms,"proxy_overhead_ms":proxy_overhead_ms,
                                "fast_ms":fast_ms,"strong_ms":strong_ms,
                                "quality":score_gold(req,selected,case.get("gold"))
                            }
                            stream.write(json.dumps(measure,allow_nan=False)+"\n")
                            stream.flush()
                            samples.append(measure)
        summary={
            "schema":"cascadeone-summary-v0.1","phase":phase,
            "arms":summarize(samples,phase),"telemetry":monitor.summary,
            "by_domain":{d:summarize([v for v in samples if v["domain"]==d],phase) for d in sorted({v["domain"] for v in samples})},
            "latency_paths":["client_http_end_to_end_ms","engine_internal_ms","proxy_overhead_ms","fast_upstream_http_ms","strong_upstream_http_ms"],
            "limitations":["sequential only","no statistical CI on p95",
                           "internal engine time includes upstream HTTP, not pure GPU compute",
                           "proxy overhead is the difference of two observations and may be slightly negative",
                           "host-wide CPU/RAM/GPU telemetry is not per-process"],
        }
        (dest/"environment.json").write_text(json.dumps({
            "platform":platform.platform(),"python":sys.version,"cpu_count":os.cpu_count(),
            "gpu_query":monitor.gpu_description},indent=2)+"\n",encoding="utf-8")
        (dest/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        checks={f.name:sha(f) for f in dest.iterdir() if f.is_file()}
        (dest/"SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name,digest in sorted(checks.items())),encoding="utf-8")
        return summary
    except Exception as e:
        (dest/"failure.json").write_text(json.dumps({"error":str(e),"kind":type(e).__name__,
                                                         "completed_trials":len(samples)},indent=2)+"\n")
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup",type=Path,help="Benchmark-owned dataset adapter/selection and cascade config")
    p.add_argument("--cascade",type=Path)
    group=p.add_mutually_exclusive_group()
    group.add_argument("--dataset",type=Path)
    group.add_argument("--dataset-id")
    p.add_argument("--adapter",default="typed_decisions_v1")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--repeats",type=int,default=2)
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--phase",choices=("colocated","strong-alone"),default="colocated")
    p.add_argument("--inference-url",default=os.environ.get("CASCADEONE_INFERENCE_URL","http://inference:18358"))
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    identity=None
    if args.setup:
        if args.cascade or args.dataset or args.dataset_id or args.adapter!="typed_decisions_v1":
            p.error("--setup may not be combined with separate --cascade/--dataset/--adapter")
        setup=json.loads(args.setup.read_text(encoding="utf-8"))
        if setup.get("schema")!="cascadeone-benchmark-setup-v1":
            raise ValueError("invalid benchmark setup schema")
        cascade=root/setup["cascade"]
        adapter=setup["adapter"]
        selection=setup.get("selection",{"type":"all"})
        source=setup["dataset"]
        sys.path.insert(0,str(root/"datasets"))
        if "id" in source:
            from catalog import resolve
            dataset,identity=resolve(source["id"])
        else:
            dataset=root/source["path"]
            identity={"mode":"local_unregistered","id":None}
        setup_identity={"id":setup["id"],"adapter":adapter,
                        "selection":selection,"config_sha256":sha(args.setup)}
    else:
        if args.cascade is None or (args.dataset is None and args.dataset_id is None):
            p.error("either --setup or --cascade with --dataset/--dataset-id is required")
        cascade=args.cascade
        dataset=args.dataset
        adapter=args.adapter
        selection={"type":"all"}
        setup_identity={"id":None,"adapter":adapter,"selection":selection}
        if args.dataset_id:
            sys.path.insert(0,str(root/"datasets"))
            from catalog import resolve
            dataset,identity=resolve(args.dataset_id)
    print(json.dumps(run(cascade,dataset,args.output,args.repeats,args.warmup,
                         phase=args.phase,dataset_identity=identity,
                         adapter=adapter,selection=selection,
                         setup_identity=setup_identity,inference_url=args.inference_url),indent=2))


if __name__=="__main__":
    main()
