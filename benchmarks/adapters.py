"""Benchmark-setup-specific dataset-to-SystemOne adapters and selection.

This module belongs to the benchmark setup, never the source dataset registry.
Offline source normalization is recorded but not included in timed inference.
"""
import hashlib
from collections import Counter
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
    return {k:v for k,v in row.items() if k!='__source_row_number'}

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


class SkipGold(Exception):
    """Source example cannot be scored without inventing a Gold target."""


def original_gold(questions, raw):
    """Complete categorical/Bernoulli targets only; never impute score moments."""
    if not isinstance(raw,dict) or set(raw)!=set(questions):
        raise ValueError("source questions and Gold mismatch")
    gold={}
    for qid,q in questions.items():
        value=raw[qid]
        kind=q["type"]
        if kind=="noul":
            if type(value) is bool:
                p=float(value)
            elif type(value) in (int,float):
                p=float(value)
            elif isinstance(value,dict) and "noul" in value:
                p=float(value["noul"])
            elif isinstance(value,dict) and "probabilities" in value:
                gold[qid]=value["probabilities"]
                p=None
            else:
                raise SkipGold("incomplete_or_unsupported_noul")
            if p is not None:
                if not math.isfinite(p) or not 0<=p<=1: raise ValueError("invalid Bernoulli Gold")
                gold[qid]={"false":1-p,"true":p}
        elif kind in ("choice","score"):
            if isinstance(value,dict) and "probabilities" in value:
                gold[qid]=value["probabilities"]
            elif isinstance(value,dict) and ("mean" in value or "variance" in value):
                raise SkipGold("score_moments_not_categorical")
            else:
                if isinstance(value,dict):
                    value=value.get("choice",value.get("score"))
                if kind=="score" and type(value) in (int,float):
                    if not float(value).is_integer():
                        raise SkipGold("continuous_score_without_distribution")
                    value=str(int(value))
                if not isinstance(value,str):
                    raise SkipGold("unsupported_categorical_gold")
                gold[qid]=value
        else:
            raise SkipGold("unsupported_question_type")
        options = (list(q["criteria"]) if kind=="choice" else
                   [str(i) for i in range(len(q["criteria"]))] if kind=="score"
                   else ["false","true"])
        truth=gold[qid]
        if isinstance(truth,str):
            if truth not in options:
                raise ValueError("Gold option not in request")
        else:
            if not set(truth).issubset(options): raise ValueError("Gold option outside request")
            if set(truth)!=set(options):
                if abs(sum(truth.values())-1)>0.0002: raise SkipGold("partial_categorical_gold")
                truth={key:float(truth.get(key,0)) for key in options}
                gold[qid]=truth
            if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1
                   for v in truth.values()) or abs(sum(truth.values())-1)>0.002:
                raise ValueError("invalid source Gold distribution")
    return gold


def helmo_gold(row):
    questions=decode(row["questions"])
    if not isinstance(questions,dict): raise ValueError("invalid helmo questions")
    return {
        "id":"helmo:"+str(row["__source_row_number"]),
        "domain":str(row.get("topic","helmo")),
        "request":{"state":decode(row["state"]),"questions":questions},
        "gold":original_gold(questions,decode(row["gold"]))
    }


def procedural_gold(row):
    questions=decode(row["questions"])
    if not isinstance(questions,dict): raise ValueError("invalid procedural questions")
    return {
        "id":"procedural:"+str(row["id"]),
        "domain":str(row.get("task","procedural")),
        "request":{"state":decode(row["state"]),"questions":questions},
        "gold":original_gold(questions,decode(row["answers"]))
    }


def hellaswag_gold(row):
    options=row["endings"]
    if not isinstance(options,list) or len(options)!=4:
        raise ValueError("HellaSwag requires four endings")
    label=str(row["label"])
    if label not in ("0","1","2","3"):
        raise SkipGold("unlabeled_hellaswag")
    return {
        "id":"hellaswag:"+str(row["ind"])+":"+str(row["__source_row_number"]),
        "domain":"hellaswag:"+str(row.get("activity_label") or "unknown"),
        "request":{"state":row["ctx"],"questions":{"continuation":{
            "type":"choice","instructions":"Select the most plausible continuation of the given passage.",
            "criteria":{str(i):value for i,value in enumerate(options)}
        }}},
        "gold":{"continuation":label}
    }


def source_rows(path):
    """Parquet readout happens in benchmark adapter, not the dataset catalog."""
    path=Path(path)
    if path.suffix==".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet benchmarks require pyarrow==21.0.0") from exc
        i=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                row["__source_row_number"]=i
                i+=1
                yield row
    else:
        with path.open(encoding="utf-8") as src:
            for i,line in enumerate(src):
                if line.strip():
                    row=json.loads(line)
                    row["__source_row_number"]=i
                    yield row


ADAPTERS={"typed_decisions_v1":typed_decisions,"n4ze3m_gold_v1":n4ze3m_gold,
          "helmo_gold_v1":helmo_gold,"procedural_gold_v1":procedural_gold,
          "hellaswag_gold_v1":hellaswag_gold}


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


def normalize(source_path,adapter="typed_decisions_v1",selection=None,with_audit=False):
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
    audit=Counter()
    for raw in source_rows(source_path):
        audit["source_rows"]+=1
        try:
            row=ADAPTERS[adapter](raw)
        except SkipGold as exc:
            audit["excluded_"+str(exc)]+=1
            continue
        key=row["id"]
        if not isinstance(key,str) or not key or key in seen:
            raise ValueError("invalid/duplicate normalized ID")
        seen.add(key)
        if not choose(row,selection):
            audit["not_selected"]+=1
            continue
        digest.update(canonical(row).encode("utf-8")+b"\n")
        output.append(row)
        audit["selected"]+=1
    if not output: raise ValueError("benchmark setup selected no rows")
    if with_audit:
        return output,digest.hexdigest(),dict(audit)
    return output,digest.hexdigest()
