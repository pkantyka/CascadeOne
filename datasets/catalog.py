"""Source dataset registry: acquisition, checksums and licenses; no benchmark processing.

Dataset-to-request conversion, Gold mapping, filtering and sampling belong to the
benchmark setup (see benchmarks/adapters.py), not to this catalog.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parent
REGISTRY=ROOT/"registry.json"
CACHE=ROOT/"cache"
SCHEMA="cascadeone-dataset-registry-v1"

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()

def safe_relative(value):
    if not isinstance(value,str) or not value or "\\" in value:
        raise ValueError("invalid relative source path")
    p=Path(value)
    if p.is_absolute() or any(part in (".","..","") for part in p.parts):
        raise ValueError("unsafe source path")
    return p

def registry(path=REGISTRY):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema")!=SCHEMA or not isinstance(data.get("datasets"),list):
        raise ValueError("invalid source registry")
    result={}
    for entry in data["datasets"]:
        key=entry["id"]
        if not re.fullmatch("[a-z0-9][a-z0-9-]*",key) or key in result:
            raise ValueError("duplicate/invalid source ID")
        if entry["mode"] not in ("downloadable","bundled"):
            raise ValueError("unsupported delivery mode")
        result[key]=entry
    return result

def get(dataset_id,registry_path=REGISTRY):
    return registry(registry_path)[dataset_id]

def hf_url(source):
    if source.get("provider")!="huggingface": raise ValueError("unsupported source provider")
    repo,revision=source["repo"],source["revision"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",repo):
        raise ValueError("invalid dataset repository ID")
    if not re.fullmatch("[a-f0-9]{40}",revision): raise ValueError("unpinned dataset revision")
    rel=str(safe_relative(source["file"]))
    return "https://huggingface.co/datasets/"+repo+"/resolve/"+revision+"/"+urllib.parse.quote(rel,safe="/")

def resolve(dataset_id,registry_path=REGISTRY,cache=CACHE,download=False,opener=None):
    entry=get(dataset_id,registry_path)
    source=entry["source"]
    if entry["mode"]=="bundled":
        root=Path(registry_path).resolve().parent
        path=(root/safe_relative(source["path"])).resolve()
        if root not in path.parents: raise ValueError("bundled path escapes registry root")
    else:
        path=Path(cache)/"raw"/entry["id"]/safe_relative(source["file"])
        if not path.is_file() and download:
            if opener is None: opener=urllib.request.urlopen
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_name(path.name+".partial")
            if temp.exists(): raise FileExistsError("incomplete previous download")
            headers={"User-Agent":"CascadeOne-Community/0.2"}
            token=os.getenv("HF_TOKEN")
            if token: headers["Authorization"]="Bearer "+token
            try:
                with opener(urllib.request.Request(hf_url(source),headers=headers),timeout=120) as response, temp.open("xb") as stream:
                    shutil.copyfileobj(response,stream,length=1024*1024)
                if sha(temp)!=source["sha256"]: raise ValueError("source SHA mismatch")
                temp.rename(path)
            except Exception:
                temp.unlink(missing_ok=True)
                raise
    if not path.is_file(): raise FileNotFoundError(f"source not available; acquire {dataset_id} first")
    if sha(path)!=source["sha256"]: raise ValueError("source SHA mismatch")
    return path,{
        "id":dataset_id,"mode":entry["mode"],"domain":entry.get("domain"),
        "license_status":entry.get("license_status"),"source_sha256":source["sha256"],
        "registry_sha256":sha(registry_path)
    }

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="cmd",required=True)
    subs.add_parser("list")
    for action in ("acquire","verify","path"):
        subs.add_parser(action).add_argument("dataset_id")
    opts=parser.parse_args()
    if opts.cmd=="list":
        print(json.dumps([{"id":x["id"],"mode":x["mode"],"domain":x.get("domain")}
                          for x in registry().values()],indent=2))
    else:
        path,identity=resolve(opts.dataset_id,download=opts.cmd=="acquire")
        print(json.dumps({"path":str(path),"identity":identity},indent=2))

if __name__=="__main__":
    main()
