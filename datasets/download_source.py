"""Acquire one complete pinned source snapshot (all registered raw split files).

Acquisition never filters, maps Gold, samples, or removes duplicate records.
Source layouts are preserved in registry source paths and cache/raw/<id>/.
"""
import argparse
import json
from pathlib import Path

from catalog import get, resolve

SOURCE_ROOT=Path(__file__).resolve().parent/"sources"


def names():
    return sorted(p.parent.name for p in SOURCE_ROOT.glob("*/manifest.json"))


def manifest(name):
    if name not in names():
        raise ValueError("unknown source group: "+name)
    data=json.loads((SOURCE_ROOT/name/"manifest.json").read_text(encoding="utf-8"))
    if data.get("schema")!="cascadeone-raw-source-group-v1" or data.get("name")!=name:
        raise ValueError("invalid source manifest")
    if not isinstance(data.get("files"),list) or not data["files"]:
        raise ValueError("empty source manifest")
    upstream=data["upstream"]
    seen=set()
    for item in data["files"]:
        ident=item["id"]
        if ident in seen: raise ValueError("duplicate source file ID")
        seen.add(ident)
        source=get(ident)["source"]
        if (source["provider"]!=upstream["provider"] or
            source["repo"]!=upstream["repository"] or
            source["revision"]!=upstream["revision"] or
            source["file"]!=item["path"] or
            source["sha256"]!=item["sha256"]):
            raise ValueError("source registry drift: "+ident)
    return data


def acquire(name,verify=False,only=None):
    data=manifest(name)
    if only and only not in {f["id"] for f in data["files"]}:
        raise ValueError("source ID not part of requested group")
    items=[f for f in data["files"] if only is None or only==f["id"]]
    for item in items:
        file,identity=resolve(item["id"],download=not verify)
        print(json.dumps({"source":name,"id":item["id"],
                          "file":str(file),"sha256":identity["source_sha256"]}),
              flush=True)
    return len(items)


def main(group=None):
    parser=argparse.ArgumentParser(description=__doc__)
    if group is None:
        parser.add_argument("source",choices=names())
    parser.add_argument("--verify",action="store_true",help="only verify cached raw bytes")
    parser.add_argument("--list",action="store_true",help="list source members without downloads")
    parser.add_argument("--dataset-id",help="acquire just one raw source member")
    args=parser.parse_args()
    name=group or args.source
    if args.list:
        print(json.dumps(manifest(name),indent=2,ensure_ascii=False))
    else:
        acquire(name,verify=args.verify,only=args.dataset_id)


if __name__=="__main__":
    main()
