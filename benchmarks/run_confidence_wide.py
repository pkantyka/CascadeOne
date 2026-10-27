"""Run groups of semantically named cascade benchmarks with three arms.

Compare Pico-only, Ultra-only and Pico->confidence->Ultra on every setup.
No G1, re-training, label-based threshold choice or cached timing.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"datasets"))
sys.path.insert(0,str(ROOT/"tools"))
from catalog import resolve  # noqa: E402
from identity import benchmark_id  # noqa: E402
from runner import run, load_dataset, sha  # noqa: E402
from validate_result import validate  # noqa: E402

def benchmark_setups(suite):
    """One folder = one source/split + model pair + routing strategy."""
    setups=sorted((ROOT/"benchmarks").glob("*/benchmark.json"))
    selected=[]
    ids=set()
    for path in setups:
        config=json.loads(path.read_text(encoding="utf-8"))
        ident=config.get("id")
        cascade_path=ROOT/config["cascade"]
        cascade=json.loads(cascade_path.read_text(encoding="utf-8"))
        if ident!=path.parent.name or ident!=benchmark_id(config,cascade):
            raise ValueError("benchmark ID does not match dataset, models and strategy: "+str(path))
        if ident in ids: raise ValueError("duplicate benchmark ID: "+ident)
        ids.add(ident)
        if config.get("suite")==suite:
            selected.append(path)
    return selected

def paired_ci(path,draws=600):
    with (path/"measurements.jsonl").open(encoding="utf-8") as stream:
        trials=[json.loads(line) for line in stream]
    byid=defaultdict(lambda:defaultdict(list))
    for r in trials:
        byid[r["id"]][r["arm"]].append(r)
    reqids=sorted(byid)
    deltas=[]
    basetimes=[]
    for key in reqids:
        arms=byid[key]
        if set(arms)!={"strong_only","fast_only","cascade"}:
            raise ValueError("incomplete paired arms")
        if any(len(arms[x])!=len(arms["cascade"]) for x in arms):
            raise ValueError("incomplete repeats")
        if any(r["quality"] is None for x in arms for r in arms[x]):
            raise ValueError("Gold required for quality CI")
        n=arms["cascade"][0]["quality"]["questions"]
        if any(r["quality"]["questions"]!=n for x in arms for r in arms[x]):
            raise ValueError("question count mismatch")
        q={x:statistics.mean(r["quality"]["question_expected_accuracy"] for r in arms[x])
           for x in arms}
        deltas.append((n*(q["cascade"]-q["strong_only"]),
                       n*(q["cascade"]-q["fast_only"]),n))
        basetimes.append(tuple(sum(r["latency_ms"] for r in arms[x])/len(arms[x])
                         for x in ("cascade","strong_only")))
    def metrics(indices):
        qs=sum(deltas[i][2] for i in indices)
        c=sum(deltas[i][0] for i in indices)/qs*100
        f=sum(deltas[i][1] for i in indices)/qs*100
        speed=100*(1-sum(basetimes[i][0] for i in indices)/
                     sum(basetimes[i][1] for i in indices))
        return c,f,speed
    observed=metrics(range(len(reqids)))
    rng=random.Random(20261027)
    replicas=[metrics([rng.randrange(len(reqids)) for _ in reqids]) for _ in range(draws)]
    ci=[]
    for j in range(3):
        values=sorted(row[j] for row in replicas)
        ci.append([values[int(.025*(draws-1))],values[int(.975*(draws-1))]])
    return {"paired_request_bootstrap_draws":draws,
            "delta_quality_pp_cascade_vs_ultra":observed[0],
            "delta_quality_pp_cascade_vs_pico":observed[1],
            "speedup_pct_cascade_vs_ultra":observed[2],
            "ci95_quality_pp_vs_ultra":ci[0],
            "ci95_quality_pp_vs_pico":ci[1],
            "ci95_speedup_pct_vs_ultra":ci[2]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",type=Path)
    p.add_argument("--suite",choices=("wide","external","source","smoke"),default="wide")
    p.add_argument("--setup-id",help="run one setup from the selected suite")
    p.add_argument("--repeats",type=int,default=2)
    p.add_argument("--warmup",type=int,default=3)
    # A benchmark always runs all three arms. Strong-alone is separate
    # auxiliary residency analysis available only through runner.py.
    p.add_argument("--dry-run",action="store_true",help="verify/normalize datasets without running models")
    p.add_argument("--list",action="store_true",
                   help="list named three-arm benchmarks without acquiring data")
    a=p.parse_args()
    if a.repeats<1 or a.warmup<0:
        p.error("invalid repeats/warmup")
    if a.output_root is None:
        a.output_root=ROOT/("output/confidence-"+a.suite)
    setups=benchmark_setups(a.suite)
    if a.setup_id:
        setups=[path for path in setups
                if json.loads(path.read_text(encoding="utf-8"))["id"]==a.setup_id]
    if not setups:
        raise ValueError("no matching benchmark setup")
    if a.list:
        for spath in setups:
            setup=json.loads(spath.read_text(encoding="utf-8"))
            print(json.dumps({"benchmark":setup["id"],
                  "dataset":setup["dataset"]["id"],
                  "cascade":setup["cascade"],
                  "arms":["fast_only","strong_only","cascade"]}))
        return
    if not a.dry_run and a.output_root.exists():
        raise FileExistsError("suite output exists, refusing overwrite")
    report=[]
    for spath in setups:
        setup=json.loads(spath.read_text(encoding="utf-8"))
        cascade_config=json.loads((ROOT/setup["cascade"]).read_text(encoding="utf-8"))
        # Three-arm comparison is mandatory; no special-case single-model
        # benchmark exists. Only the cascade arm uses routing strategy.
        if "fast" not in cascade_config or "strong" not in cascade_config or "routing" not in cascade_config:
            raise ValueError("benchmark must specify two models and a routing strategy")
        dataset,identity=resolve(setup["dataset"]["id"])
        selection=setup.get("selection",{"type":"all"})
        adapter=setup["adapter"]
        if a.dry_run:
            rows,digest,audit=load_dataset(dataset,adapter,selection,with_audit=True)
            report.append({"setup":setup["id"],"rows":len(rows),"normalized_sha256":digest,
                           "source_sha256":identity["source_sha256"],"normalization":audit})
            print(json.dumps(report[-1]),flush=True)
            continue
        run_path=a.output_root if a.setup_id else a.output_root/setup["id"]
        result=run(ROOT/setup["cascade"],dataset,run_path,a.repeats,a.warmup,
                   phase="colocated",dataset_identity=identity,adapter=adapter,
                   selection=selection,setup_identity={"id":setup["id"],
                       "adapter":adapter,"selection":selection,"config_sha256":sha(spath)})
        verified=validate(run_path)
        required_arms={"fast_only","strong_only","cascade"}
        if set(result["arms"])!=required_arms:
            raise ValueError("benchmark must measure all three arms")
        item={"setup":setup["id"],"source":identity,"verified":verified,
              "arms":result["arms"],"phase":"colocated"}
        item["paired"]=paired_ci(run_path)
        report.append(item)
        print(json.dumps({"setup":item["setup"],"paired":item.get("paired")}),flush=True)
    if not a.dry_run:
        a.output_root.mkdir(parents=True,exist_ok=True)
        (a.output_root/"suite.json").write_text(json.dumps({
            "kind":"cascadeone-benchmark-suite-"+a.suite+"-v1","status":"complete",
            "required_arms":["fast_only","strong_only","cascade"],
            "selection":"frozen per-setup; no calibration on test",
            "source_warning":{
                "wide":"train sources; model pretraining contamination not ruled out",
                "external":"public test/heldout; model training overlap unknown; BJB banking77 and clinc150 overlap originals",
                "source":"full upstream n4ze3m train split; model pretraining overlap not ruled out",
                "smoke":"synthetic integration-only data; not representative of model quality"
            }[a.suite],
            "runs":report},ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"setups":len(report),"dry_run":a.dry_run},indent=2))

if __name__=="__main__":
    main()
