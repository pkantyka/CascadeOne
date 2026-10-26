"""Compare two runs only when their cascade and dataset fingerprints match."""
import argparse
import json
from pathlib import Path

def read(root):
    root=Path(root)
    return json.loads((root/"manifest.json").read_text()),json.loads((root/"summary.json").read_text())

def diff(reference,other):
    if reference is None or other is None: return None
    if reference==0: return None
    return (other/reference-1)*100

def compare(solo,paired):
    m0,s0=read(solo)
    m1,s1=read(paired)
    for key in ("cascade_id","cascade_config_sha256","dataset_sha256","repeats"):
        if m0[key]!=m1[key]: raise ValueError("incompatible runs: "+key)
    if m0["phase"]!="strong-alone" or m1["phase"]!="colocated":
        raise ValueError("expected strong-alone then colocated")
    e0=json.loads((Path(solo)/"environment.json").read_text())
    e1=json.loads((Path(paired)/"environment.json").read_text())
    for key in ("platform","cpu_count","gpu_query"):
        if e0.get(key)!=e1.get(key): raise ValueError("hardware/environment mismatch: "+key)
    b=s1["arms"]["strong_only"]
    c=s1["arms"]["cascade"]
    perf=s0["arms"]["strong_only"]
    out={
      "schema":"cascadeone-comparison-v0.1",
      "cascade_id":m1["cascade_id"],
      "colocated_strong_mean_ms":b["mean_ms"],
      "colocated_cascade_mean_ms":c["mean_ms"],
      "mean_latency_reduction_vs_colocated_strong":1-c["mean_ms"]/b["mean_ms"],
      "p95_latency_ratio_vs_colocated_strong":c["p95_ms"]/b["p95_ms"],
      "gold_delta_pp":c["gold_delta_pp_vs_strong_only"],
      "strong_alone_mean_ms":perf["mean_ms"],
      "strong_alone_p95_ms":perf["p95_ms"],
      "host_telemetry":{}
    }
    for key in ("host_ram_used_mib","vram_used_mib","cpu_util_percent",
                "gpu_util_percent","gpu_power_watts"):
        a=s0["telemetry"].get(key)
        z=s1["telemetry"].get(key)
        out["host_telemetry"][key]={
          "standalone_peak":a["peak"] if a else None,
          "colocated_peak":z["peak"] if z else None,
          "peak_relative_change_percent":diff(a["peak"],z["peak"]) if a and z else None,
          "scope":"host-wide"
        }
    out["warning"]="Memory/CPU/GPU telemetry is host-wide; phases use different service residency. Not process attribution."
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--standalone",required=True)
    p.add_argument("--colocated",required=True)
    p.add_argument("--output")
    opts=p.parse_args()
    value=json.dumps(compare(opts.standalone,opts.colocated),indent=2)+"\n"
    if opts.output: Path(opts.output).write_text(value)
    print(value)
