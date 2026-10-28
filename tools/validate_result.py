"""Validate a submitted Community run against its hashes and minimal v0.1 contract."""
import argparse
import hashlib
import json
import math
from pathlib import Path

REQUIRED=("manifest.json","environment.json","measurements.jsonl","telemetry.jsonl","summary.json","SHA256SUMS")

def validate(root):
    root=Path(root)
    if not root.is_dir(): raise ValueError("missing run directory")
    for name in REQUIRED:
        if not (root/name).is_file(): raise ValueError("missing "+name)
    lines=(root/"SHA256SUMS").read_text().splitlines()
    hashes={}
    for line in lines:
        h,name=line.split("  ",1)
        if name in hashes or name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("invalid SHA256SUMS")
        hashes[name]=h
    if set(hashes)!=set(REQUIRED)-{"SHA256SUMS"}:
        raise ValueError("unexpected or missing indexed file")
    for name,expected in hashes.items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual!=expected: raise ValueError("SHA mismatch: "+name)
    manifest=json.loads((root/"manifest.json").read_text())
    summary=json.loads((root/"summary.json").read_text())
    if manifest.get("schema")!="cascadeone-run-v0.1" or summary.get("schema")!="cascadeone-summary-v0.1":
        raise ValueError("unsupported result version")
    if manifest.get("status")!="complete" or manifest.get("repeats",0)<1 or manifest.get("n_requests",0)<1:
        raise ValueError("invalid run status/counts")
    phase=manifest.get("phase")
    arms=set(summary["arms"])
    if arms!=({"strong_only"} if phase=="strong-alone" else {"strong_only","fast_only","cascade"}):
        raise ValueError("missing benchmark arms")
    trials=[json.loads(x) for x in (root/"measurements.jsonl").read_text().splitlines()]
    if manifest.get("execution") == "systemone_inference_proxy_http":
        if manifest.get("engine_version") != "inference-systemone-v1":
            raise ValueError("missing shared inference engine version")
        for arm in arms:
            metrics = summary["arms"][arm]
            for key in ("mean_ms", "p95_ms", "engine_mean_ms", "engine_p95_ms",
                        "proxy_overhead_mean_ms", "proxy_overhead_p95_ms"):
                value = metrics.get(key)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("missing or invalid dual latency summary: " + key)
        for trial in trials:
            for key in ("latency_ms", "engine_ms", "proxy_overhead_ms", "fast_ms", "strong_ms"):
                value = trial.get(key)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("missing or invalid dual latency trial: " + key)
            if min(trial["latency_ms"], trial["engine_ms"]) <= 0 or min(trial["fast_ms"],trial["strong_ms"]) < 0:
                raise ValueError("invalid nonpositive latency")
            if abs(trial["proxy_overhead_ms"] - (trial["latency_ms"] - trial["engine_ms"])) > 1e-6:
                raise ValueError("invalid proxy overhead decomposition")
    expected=manifest["n_requests"]*manifest["repeats"]*len(arms)
    if len(trials)!=expected: raise ValueError("missing measurements")
    keys={(r["id"],r["repeat"],r["arm"]) for r in trials}
    if len(keys)!=len(trials): raise ValueError("duplicate trial")
    if not isinstance(manifest.get("routing"),dict) or not manifest["routing"].get("type"):
        raise ValueError("missing declared routing policy")
    if not isinstance(manifest.get("benchmark_setup"),dict) or not manifest["benchmark_setup"].get("adapter"):
        raise ValueError("missing benchmark adapter")
    normalized=manifest.get("normalized_requests_sha256")
    if not isinstance(normalized,str) or len(normalized)!=64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError("missing/invalid normalized input hash")
    identity=manifest.get("dataset_identity",{})
    if identity.get("source_sha256") is not None and identity["source_sha256"]!=manifest["dataset_sha256"]:
        raise ValueError("reported source dataset hash mismatch")
    for trial in trials:
        route=trial.get("routing")
        if not isinstance(route,dict) or route.get("escalated") is not trial.get("escalated"):
            raise ValueError("routing decision mismatch")
        expected_policy=trial["arm"] if trial["arm"] in ("strong_only","fast_only") else manifest["routing"]["type"]
        if route.get("policy")!=expected_policy:
            raise ValueError("routing policy mismatch")
        if not isinstance(route.get("signals"),dict):
            raise ValueError("routing signals must be an object")
    for arm in arms:
        if sum(r["arm"]==arm for r in trials)!=manifest["n_requests"]*manifest["repeats"]:
            raise ValueError("incomplete arm")
    return {"valid":True,"trials":len(trials),"phase":phase}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    print(json.dumps(validate(p.parse_args().run),indent=2))
