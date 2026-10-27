"""Canonical human-readable ID for a three-arm cascade benchmark.

Every benchmark directory name = dataset/split (and any benchmark-only
selection) + model pair + routing strategy. Exact precision is in configs.
"""
import re

PAIRS={
    ("Remek/basal-omni-pico","Remek/basal-omni-1.0-ultra"):"basal-pico-ultra",
    ("convaiinnovations/laya-typed-decisions","Cloudflare/clef-flash"):"laya-clef",
}
SPECIAL_DATASETS={
    "smoke-general-v1":"smoke-general",
    "n4ze3m-typed-decisions-v1":"n4ze3m-typed-decisions-train",
    "helmo-synthetic-v1":"helmo-synthetic-train",
    "procedural-typed-0-v1":"procedural-typed-train-shard0",
    "procedural-typed-1-v1":"procedural-typed-train-shard1",
}


def dataset_slug(source_id,selection):
    if source_id in SPECIAL_DATASETS:
        name=SPECIAL_DATASETS[source_id]
    elif source_id.startswith("bjb-") and source_id.endswith("-heldout-v1"):
        name="better-jev-bench-"+source_id[4:-11]+"-heldout"
    elif source_id.endswith("-v1"):
        name=source_id[:-3]
    else:
        raise ValueError("unknown dataset ID format")
    kind=selection.get("type","all")
    if kind=="all":
        return name
    if kind=="hash_range":
        lower=selection["start"]
        upper=selection["end"]
        seed=selection["seed"]
        return name+"-hash%06d-%06d-seed%d"%(
            round(lower*1_000_000),round(upper*1_000_000),seed)
    raise ValueError("unknown benchmark selection")


def pair_slug(cascade):
    small=cascade["fast"]["model_id"]
    large=cascade["strong"]["model_id"]
    if (small,large) in PAIRS:
        return PAIRS[(small,large)]
    def clean(name):
        return re.sub("[^a-z0-9]+","-",name.split("/")[-1].lower()).strip("-")
    if small==large: raise ValueError("small and large models must differ")
    return clean(small)+"-to-"+clean(large)


def strategy_slug(cascade):
    routing=cascade["routing"]
    name=routing["type"]
    params=routing.get("parameters",{})
    if name=="confidence_threshold":
        agg={"minimum":"min","mean":"mean","maximum":"max"}[
            params.get("aggregation","minimum")]
        threshold=float(params["threshold"])
        if not 0<=threshold<=1:
            raise ValueError("invalid confidence threshold")
        return "confidence-"+agg+"-t"+("%.6f"%threshold).replace(".","p")
    if name=="max_questions_pre_router":
        return "pre-router-max-questions-"+str(params["max_questions"])
    if name in ("always_fast","always_strong"):
        return name.replace("_","-")
    if name=="plugin":
        return "plugin-"+re.sub("[^a-z0-9]+","-",routing["factory"].lower()).strip("-")
    raise ValueError("unknown strategy slug")


def benchmark_id(setup,cascade):
    return (dataset_slug(setup["dataset"]["id"],setup.get("selection",{"type":"all"}))
            +"__"+pair_slug(cascade)+"__"+strategy_slug(cascade))
