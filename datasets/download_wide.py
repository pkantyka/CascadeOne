"""Acquire or verify pinned raw benchmark sources. No conversion or selection."""
import argparse
import json
from pathlib import Path
from catalog import get, resolve

ROOT=Path(__file__).resolve().parents[1]
SETUPS=sorted((ROOT/"benchmarks/setups").glob("pico-ultra-wide-*.json"))

def dataset_ids():
    return sorted({json.loads(p.read_text())["dataset"]["id"] for p in SETUPS})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify",action="store_true",help="verify existing files without network")
    p.add_argument("--dataset-id",action="append",choices=dataset_ids())
    a=p.parse_args()
    for ident in a.dataset_id or dataset_ids():
        path,identity=resolve(ident,download=not a.verify)
        print(json.dumps({"id":ident,"file":str(path),"sha256":identity["source_sha256"],
            "repo":get(ident)["source"]["repo"]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
