"""Run independent H1 baselines on chosen complete BJB held-out text tasks.

Multimodal source files remain downloadable, but cannot be benchmarked by
the text-only adapter without sourcing and verifying referenced images.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
TASKS=("banking77","clinc150","cfpb","cuad","civil-comments",
       "go-emotions","ledgar","massive")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",action="append",choices=TASKS,
                   help="one or more tasks; default all supported text tasks")
    p.add_argument("--output-root",type=Path,default=Path("output/benchmarks/better-jev-bench"))
    p.add_argument("--repeats",type=int,default=2)
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    if a.repeats<1 or a.warmup<0:
        p.error("invalid repeats/warmup")
    for name in dict.fromkeys(a.task or TASKS):
        cmd=[sys.executable,str(ROOT/"benchmarks/run_confidence_wide.py"),
             "--suite","external","--setup-id","pico-ultra-external-bjb-"+name,
             "--output-root",str(a.output_root/name),
             "--repeats",str(a.repeats),"--warmup",str(a.warmup)]
        if a.dry_run:
            cmd.append("--dry-run")
        print("RUN:",name,flush=True)
        subprocess.run(cmd,cwd=ROOT,check=True)


if __name__=="__main__":
    main()
