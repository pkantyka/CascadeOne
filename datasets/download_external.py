"""Download or SHA-256 verify pinned external System One benchmark raw files.

Use the existing source catalog: no dataset conversion or train/test splitting.
See benchmarks/setups/pico-ultra-external-*.json for independent setups.
"""
import argparse
import json
from pathlib import Path
from catalog import resolve

ROOT=Path(__file__).resolve().parents[1]
SETUPS=sorted((ROOT/"benchmarks/setups").glob("pico-ultra-external-*.json"))


def identities():
    return sorted({json.loads(path.read_text(encoding="utf-8"))["dataset"]["id"]
                   for path in SETUPS})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true",
                        help="check cached source files without network")
    parser.add_argument("--list",action="store_true",
                        help="list source IDs; no downloads")
    parser.add_argument("--dataset-id",action="append",
                        choices=identities(),help="download selected ID(s)")
    args=parser.parse_args()
    chosen=args.dataset_id or identities()
    if args.list:
        print(json.dumps(chosen,indent=2))
        return
    for name in chosen:
        source,provenance=resolve(name,download=not args.verify)
        print(json.dumps({"id":name,"file":str(source),
                          "sha256":provenance["source_sha256"],
                          "registry_sha256":provenance["registry_sha256"]}),
              flush=True)


if __name__=="__main__":
    main()
