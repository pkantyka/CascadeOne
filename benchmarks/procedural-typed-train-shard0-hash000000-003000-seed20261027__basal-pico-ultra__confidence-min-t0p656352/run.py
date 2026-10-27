"""Execute this benchmark, comparing small, large and cascade."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
if __name__=="__main__":
    cmd=[sys.executable,str(ROOT/"benchmarks/run_confidence_wide.py"),
         "--suite","wide","--setup-id","procedural-typed-train-shard0-hash000000-003000-seed20261027__basal-pico-ultra__confidence-min-t0p656352",
         "--output-root",str(ROOT/"output/benchmarks"/"procedural-typed-train-shard0-hash000000-003000-seed20261027__basal-pico-ultra__confidence-min-t0p656352")]+sys.argv[1:]
    raise SystemExit(subprocess.call(cmd,cwd=ROOT))
