"""Versioned two-mode benchmark datasets: bundled files or pinned HF acquisition.

Standard library only. Download/processing is explicit; never redistributes
raw licensed files. Fully prepared datasets are immutable and SHA-checked.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "registry.json"
CACHE = ROOT / "cache"
SCHEMA = "cascadeone-dataset-registry-v1"
PREPARED_SCHEMA = "cascadeone-prepared-dataset-v1"
ADAPTER = "n4ze3m-typed-gold-v1"

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()

def safe_relative(value):
    if not isinstance(value,str) or not value or "\\" in value:
        raise ValueError("invalid relative file path")
    p=Path(value)
    if p.is_absolute() or any(x in ("..",".","") for x in p.parts):
        raise ValueError("unsafe relative path")
    return p

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"))

def registry(path=REGISTRY):
    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema")!=SCHEMA:
        raise ValueError("unrecognized dataset registry schema")
    ids=[entry["id"] for entry in payload["datasets"]]
    if len(set(ids))!=len(ids):
        raise ValueError("duplicate dataset ID")
    for entry in payload["datasets"]:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*",entry["id"]):
            raise ValueError("invalid dataset ID")
        if entry["mode"] not in ("bundled","downloadable"):
            raise ValueError("unknown distribution mode")
    return {x["id"]:x for x in payload["datasets"]}

def get(dataset_id,registry_path=REGISTRY):
    data=registry(registry_path)
    if dataset_id not in data: raise KeyError("unknown dataset: "+dataset_id)
    return data[dataset_id]

def validate_rows(path):
    seen=set()
    n=0
    gold_count=0
    domains=Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row=json.loads(line)
        key=row["id"]
        if not isinstance(key,str) or not key or key in seen:
            raise ValueError("duplicate/invalid ID")
        seen.add(key)
        request=row["request"]
        if not isinstance(request.get("state"),(str,dict,list)):
            raise ValueError("invalid state")
        questions=request.get("questions")
        if not isinstance(questions,dict) or not questions:
            raise ValueError("invalid questions")
        for q in questions.values():
            kind=q.get("type")
            if kind=="choice" and (not isinstance(q.get("criteria"),dict) or not q["criteria"]):
                raise ValueError("invalid choice")
            elif kind=="score" and (not isinstance(q.get("criteria"),list) or len(q["criteria"])<2):
                raise ValueError("invalid score")
            elif kind=="noul": pass
            elif kind not in ("choice","score","noul"): raise ValueError("invalid question kind")
        gold=row.get("gold")
        if gold is not None:
            if not isinstance(gold,dict) or set(gold)!=set(questions):
                raise ValueError("incomplete Gold coverage")
            for qid,truth in gold.items():
                q=questions[qid]
                options=(set(q["criteria"]) if q["type"]=="choice" else
                         set(str(i) for i in range(len(q["criteria"]))) if q["type"]=="score" else
                         {"false","true"})
                if not isinstance(truth,str) or truth not in options:
                    raise ValueError("invalid Gold label")
            gold_count+=1
        n+=1
        domains[row.get("domain","unspecified")]+=1
    if n==0: raise ValueError("empty prepared dataset")
    return {"rows":n,"gold_requests":gold_count,"domains":dict(sorted(domains.items())),"ids":seen}

def check_bundled(entry):
    result={}
    for split,meta in entry["splits"].items():
        path=ROOT/safe_relative(meta["path"])
        if not path.is_file() or sha(path)!=meta["sha256"]:
            raise ValueError("bundled checksum mismatch: "+split)
        checked=validate_rows(path)
        if checked["rows"]!=meta["rows"]:
            raise ValueError("bundled row count mismatch")
        result[split]={"path":path,"info":checked}
    if len(result)>1 and len(set.union(*(value["info"]["ids"] for value in result.values())))!=sum(value["info"]["rows"] for value in result.values()):
        raise ValueError("bundled split leakage")
    return result

def decode(value):
    if isinstance(value,str):
        try: return json.loads(value)
        except ValueError: return value
    return value

def convert_n4ze3m(raw):
    state=decode(raw["state"])
    questions=decode(raw["questions"])
    labels=decode(raw["gold"])
    if not isinstance(questions,dict) or not isinstance(labels,dict) or set(questions)!=set(labels):
        raise ValueError("invalid n4ze3m request/Gold")
    gold={}
    for key,question in questions.items():
        value=labels[key]
        kind=question["type"]
        if kind=="noul":
            if type(value) is not bool: raise ValueError("non-boolean Gold")
            gold[key]="true" if value else "false"
        elif kind=="score":
            if type(value) is not int: raise ValueError("invalid numeric score")
            gold[key]=str(value)
        elif kind=="choice":
            if not isinstance(value,str): raise ValueError("invalid choice Gold")
            gold[key]=value
        else: raise ValueError("unknown question type")
    result={"id":"n4ze3m:"+str(raw["state_id"]),
            "request":{"state":state,"questions":questions},
            "gold":gold,"domain":str(raw.get("domain","unspecified"))}
    return result

def split_of(item,seed):
    # Group all identical states together; neither labels nor model answers affect selection.
    state_hash=hashlib.sha256(canonical(item["request"]["state"]).encode("utf-8")).hexdigest()
    n=int(hashlib.sha256(f"{seed}:{state_hash}".encode("utf-8")).hexdigest()[:16],16)/16**16
    if n<0.20: return "calibration"
    if n<0.50: return "test"
    return "reserved"

def hf_url(source):
    if source.get("provider")!="huggingface":
        raise ValueError("unsupported download provider")
    repo=source["repo"]
    rev=source["revision"]
    rel=str(safe_relative(source["file"]))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",repo):
        raise ValueError("invalid HF repo")
    if not re.fullmatch("[a-f0-9]{40}",rev):
        raise ValueError("HF revision must be a 40-character commit")
    return "https://huggingface.co/datasets/"+repo+"/resolve/"+rev+"/"+urllib.parse.quote(rel,safe="/")

def acquire(entry,cache=CACHE,opener=None):
    source=entry["source"]
    path=Path(cache)/"raw"/entry["id"]/safe_relative(source["file"])
    if path.is_file():
        if sha(path)!=source["sha256"]:
            raise ValueError("cached raw SHA mismatch; remove manually after investigation")
        return path
    if opener is None: opener=urllib.request.urlopen
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".partial")
    if temp.exists(): raise FileExistsError("unfinished raw download; inspect/remove manually")
    headers={"User-Agent":"CascadeOne-Community/0.1"}
    token=os.getenv("HF_TOKEN")
    if token: headers["Authorization"]="Bearer "+token
    request=urllib.request.Request(hf_url(source),headers=headers)
    try:
        with opener(request,timeout=120) as response, temp.open("xb") as stream:
            shutil.copyfileobj(response,stream,length=1024*1024)
        if sha(temp)!=source["sha256"]:
            raise ValueError("downloaded raw file SHA mismatch")
        temp.rename(path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return path

def prepare_download(entry,cache=CACHE,opener=None):
    if entry.get("adapter")!=ADAPTER: raise ValueError("unrecognized conversion adapter")
    raw=acquire(entry,cache,opener)
    dest=Path(cache)/"prepared"/entry["id"]
    if dest.exists(): return verify_download(entry,cache)
    temp=dest.with_name(dest.name+".partial")
    if temp.exists(): raise FileExistsError("unfinished prepare; inspect/remove manually")
    temp.mkdir(parents=True)
    seen=set()
    counts=Counter()
    all_ids=set()
    try:
        files={split:(temp/(split+".jsonl")).open("x",encoding="utf-8",newline="\n") for split in ("calibration","test")}
        try:
            with raw.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip(): continue
                    converted=convert_n4ze3m(json.loads(line))
                    if converted["id"] in all_ids: raise ValueError("duplicate upstream ID")
                    all_ids.add(converted["id"])
                    group=hashlib.sha256(canonical(converted["request"]["state"]).encode("utf-8")).hexdigest()
                    if group in seen: raise ValueError("repeated identical state; inspect upstream")
                    seen.add(group)
                    split=split_of(converted,entry["split_seed"])
                    counts[split]+=1
                    if split in files:
                        files[split].write(json.dumps(converted,ensure_ascii=False,allow_nan=False)+"\n")
        finally:
            for file in files.values(): file.close()
        if any(counts[s]==0 for s in ("test","calibration")):
            raise ValueError("empty calibration/test partition")
        split_meta={}
        for split in ("calibration","test"):
            path=temp/(split+".jsonl")
            checked=validate_rows(path)
            split_meta[split]={"file":path.name,"sha256":sha(path),
                               "rows":checked["rows"],"gold_requests":checked["gold_requests"],
                               "domains":checked["domains"]}
        manifest={"schema":PREPARED_SCHEMA,"id":entry["id"],"mode":"downloadable",
                  "source":entry["source"],"adapter":entry["adapter"],
                  "split_seed":entry["split_seed"],"partition_counts":dict(counts),
                  "splits":split_meta,
                  "limitations":["upstream TRAIN derived splits, NOT the official independent test",
                                 "upstream Gold provenance requires review",
                                 "overlap with model pretraining may be unknown"]}
        (temp/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        temp.rename(dest)
    except Exception:
        shutil.rmtree(temp,ignore_errors=True)
        raise
    return verify_download(entry,cache)

def verify_download(entry,cache=CACHE):
    root=Path(cache)/"prepared"/entry["id"]
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema")!=PREPARED_SCHEMA or manifest.get("id")!=entry["id"] or manifest.get("source")!=entry["source"] or manifest.get("adapter")!=entry["adapter"] or manifest.get("split_seed")!=entry["split_seed"]:
        raise ValueError("prepared manifest/registry mismatch")
    raw=Path(cache)/"raw"/entry["id"]/safe_relative(entry["source"]["file"])
    if not raw.is_file() or sha(raw)!=entry["source"]["sha256"]:
        raise ValueError("missing or modified raw source")
    seen=set()
    for split,meta in manifest["splits"].items():
        if split not in ("calibration","test") or meta["file"]!=split+".jsonl":
            raise ValueError("invalid split")
        path=root/meta["file"]
        if sha(path)!=meta["sha256"]: raise ValueError("prepared SHA mismatch")
        checked=validate_rows(path)
        if checked["rows"]!=meta["rows"] or checked["gold_requests"]!=meta["gold_requests"]:
            raise ValueError("prepared population mismatch")
        if seen.intersection(checked["ids"]): raise ValueError("split ID leakage")
        seen.update(checked["ids"])
    if set(manifest["splits"])!={"calibration","test"}:
        raise ValueError("missing prepared splits")
    return {"id":entry["id"],"mode":"downloadable","manifest":root/"manifest.json",
            "splits":{split:root/meta["file"] for split,meta in manifest["splits"].items()}}

def verify(dataset_id,registry_path=REGISTRY,cache=CACHE):
    entry=get(dataset_id,registry_path)
    if entry["mode"]=="downloadable":
        return verify_download(entry,cache)
    checked=check_bundled(entry)
    return {"id":entry["id"],"mode":"bundled",
            "manifest":Path(registry_path),
            "splits":{k:v["path"] for k,v in checked.items()}}

def prepare(dataset_id,registry_path=REGISTRY,cache=CACHE,opener=None):
    entry=get(dataset_id,registry_path)
    if entry["mode"]=="downloadable":
        return prepare_download(entry,cache,opener)
    return verify(dataset_id,registry_path,cache)

def resolved(dataset_id,split="test",registry_path=REGISTRY,cache=CACHE):
    detail=verify(dataset_id,registry_path,cache)
    if split not in detail["splits"]:
        raise ValueError("unsupported dataset split: "+split)
    entry=get(dataset_id,registry_path)
    return detail["splits"][split],{
        "id":dataset_id,"mode":entry["mode"],"split":split,
        "manifest_sha256":sha(detail["manifest"]),
        "domain":entry.get("domain"),"license_status":entry.get("license_status")
    }

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="action",required=True)
    subs.add_parser("list")
    for name in ("prepare","verify","path"):
        command=subs.add_parser(name)
        command.add_argument("dataset_id")
        if name=="path":
            command.add_argument("--split",default="test")
    args=parser.parse_args()
    if args.action=="list":
        print(json.dumps([{"id":e["id"],"mode":e["mode"],"domain":e.get("domain")} for e in registry().values()],indent=2))
    elif args.action=="path":
        p,meta=resolved(args.dataset_id,args.split)
        print(json.dumps({"path":str(p),"identity":meta},indent=2))
    else:
        output=prepare(args.dataset_id) if args.action=="prepare" else verify(args.dataset_id)
        print(json.dumps({**output,"manifest":str(output["manifest"]),"splits":{k:str(v) for k,v in output["splits"].items()}},indent=2))

if __name__=="__main__":
    main()
