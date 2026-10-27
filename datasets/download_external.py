"""Acquire COMPLETE BANKING77, CLINC150, and better-jev-bench source datasets.

No source is subsetted, deduplicated, or adapted to model requests here.
"""
import argparse
import json
from catalog import resolve
from download_source import acquire, manifest, names


def identities():
    return sorted({f["id"] for name in names() for f in manifest(name)["files"]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true",help="verify cached bytes without downloading")
    parser.add_argument("--list",action="store_true",help="list all original source files")
    parser.add_argument("--source",action="append",choices=names(),
                        help="download only selected complete upstream source(s)")
    parser.add_argument("--dataset-id",action="append",choices=identities(),
                        help="optional individual raw file(s), not a full source")
    args=parser.parse_args()
    ids=args.dataset_id
    if ids and args.source:
        parser.error("--source and --dataset-id cannot be combined")
    if args.list:
        print(json.dumps({"sources":{name:[f["id"] for f in manifest(name)["files"]]
                                     for name in names()}},indent=2))
        return
    if ids:
        for ident in ids:
            path,provenance=resolve(ident,download=not args.verify)
            print(json.dumps({"id":ident,"file":str(path),
                              "sha256":provenance["source_sha256"]}),flush=True)
    else:
        for name in args.source or names():
            acquire(name,verify=args.verify)


if __name__=="__main__":
    main()
