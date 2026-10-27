"""Execute this benchmark, comparing small, large and cascade."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
if __name__=="__main__":
    cmd=[sys.executable,str(ROOT/"benchmarks/run_confidence_wide.py"),
         "--suite","smoke","--setup-id","smoke-general__laya-clef__pre-router-max-questions-1",
         "--output-root",str(ROOT/"output/benchmarks"/"smoke-general__laya-clef__pre-router-max-questions-1")]+sys.argv[1:]
    raise SystemExit(subprocess.call(cmd,cwd=ROOT))
