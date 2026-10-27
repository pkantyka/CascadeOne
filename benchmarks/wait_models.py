"""Wait for pinned Pico and Ultra services to become ready before measuring."""
import argparse
import json
from pathlib import Path
import time
from runner import check_identity, request_json

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeout-seconds",type=int,default=1800)
    args=p.parse_args()
    config=json.loads((ROOT/"cascades/basal-pico-ultra/benchmark.json").read_text())
    deadline=time.monotonic()+args.timeout_seconds
    while time.monotonic()<deadline:
        try:
            for kind in ("fast","strong"):
                endpoint=config[kind]["url"].rstrip("/")
                health=request_json(endpoint+"/health",timeout=15)
                if health.get("status")!="ok":
                    raise RuntimeError("model unhealthy: "+kind)
                check_identity(endpoint,config[kind])
            print("PICO_AND_ULTRA_READY",flush=True)
            return
        except (OSError,TimeoutError,RuntimeError,ValueError) as exc:
            print("Waiting for pinned model services:",str(exc),flush=True)
            time.sleep(10)
    raise TimeoutError("pinned models did not become ready within startup timeout")

if __name__=="__main__":
    main()
