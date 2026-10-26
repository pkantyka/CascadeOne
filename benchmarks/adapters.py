"""Benchmark-setup-specific dataset-to-SystemOne adapters and selection.

This module belongs to the benchmark setup, never the source dataset registry.
Offline source normalization is recorded but not included in timed inference.
"""
import hashlib
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)

def decode(value):
    if isinstance(value,str):
        try: return json.loads(value)
        except ValueError: return value
    return value

def typed_decisions(row):
    if not isinstance(row,dict) or "request" not in row or "id" not in row:
        raise ValueError("typed_decisions_v1 requires id and request")
    return row

def n4ze3m_gold(row):
    if not isinstance(row,dict): raise ValueError("invalid source row")
    state=decode(row["state"])
    questions=decode(row["questions"])
    labels=decode(row["gold"])
    if not isinstance(questions,dict) or not isinstance(labels,dict) or set(questions)!=set(labels):
        raise ValueError("source questions and Gold mismatch")
    gold={}
    for qid,question in questions.items():
        value=labels[qid]
        kind=question["type"]
        if kind=="noul":
            if type(value) is not bool: raise ValueError("boolean Gold expected")
            gold[qid]="true" if value else "false"
        elif kind=="score":
            if type(value) is not int: raise ValueError("integer score expected")
            gold[qid]=str(value)
        elif kind=="choice":
            if not isinstance(value,str): raise ValueError("choice Gold must be a string")
            gold[qid]=value
        else: raise ValueError("unsupported question type")
    return {"id":"n4ze3m:"+str(row["state_id"]),
            "domain":str(row.get("domain","unspecified")),
            "request":{"state":state,"questions":questions},"gold":gold}


ADAPTERS={"typed_decisions_v1":typed_decisions,"n4ze3m_gold_v1":n4ze3m_gold}


def choose(row,selection):
    kind=selection.get("type","all")
    if kind=="all": return True
    if kind=="hash_range":
        lower=selection["start"]
        upper=selection["end"]
        seed=selection["seed"]
        if (type(lower) not in (int,float) or type(upper) not in (int,float) or
            not 0<=lower<upper<=1 or type(seed) is not int or seed<0):
            raise ValueError("invalid hash_range selector")
        # Same state always selects into the same range: no exact-state leakage.
        state=canonical(row["request"]["state"])
        h=hashlib.sha256((str(seed)+":"+state).encode("utf-8")).hexdigest()
        position=int(h[:16],16)/16**16
        return lower<=position<upper
    raise ValueError("unknown benchmark selection type")


def normalize(source_path,adapter="typed_decisions_v1",selection=None):
    selection=selection or {"type":"all"}
    if not isinstance(selection,dict): raise ValueError("selection must be an object")
    if adapter not in ADAPTERS:
        raise ValueError("unrecognized benchmark setup adapter")
    # Validate selection even when all rows would otherwise be dropped.
    if selection.get("type")=="hash_range":
        choose({"request":{"state":"validation"}},selection)
    elif selection.get("type","all")!="all":
        raise ValueError("unknown benchmark selection")
    output=[]
    seen=set()
    digest=hashlib.sha256()
    for line in Path(source_path).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row=ADAPTERS[adapter](json.loads(line))
        key=row["id"]
        if not isinstance(key,str) or not key or key in seen:
            raise ValueError("invalid/duplicate normalized ID")
        seen.add(key)
        if not choose(row,selection): continue
        digest.update(canonical(row).encode("utf-8")+b"\n")
        output.append(row)
    if not output: raise ValueError("benchmark setup selected no rows")
    return output,digest.hexdigest()
