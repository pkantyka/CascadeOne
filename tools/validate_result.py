"""Validate a submitted Community run against its hashes and minimal v0.1 contract."""
import argparse
import hashlib
import json
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
    if arms!=({"strong_only"} if phase=="strong-alone" else {"strong_only","cascade"}):
        raise ValueError("missing benchmark arms")
    trials=[json.loads(x) for x in (root/"measurements.jsonl").read_text().splitlines()]
    expected=manifest["n_requests"]*manifest["repeats"]*len(arms)
    if len(trials)!=expected: raise ValueError("missing measurements")
    keys={(r["id"],r["repeat"],r["arm"]) for r in trials}
    if len(keys)!=len(trials): raise ValueError("duplicate trial")
    for arm in arms:
        if sum(r["arm"]==arm for r in trials)!=manifest["n_requests"]*manifest["repeats"]:
            raise ValueError("incomplete arm")
    return {"valid":True,"trials":len(trials),"phase":phase}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    print(json.dumps(validate(p.parse_args().run),indent=2))
