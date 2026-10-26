"""Best-effort host-level resources, never misrepresented as per-model usage."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import threading

def cpu_ticks():
    try:
        with open("/proc/stat",encoding="utf-8") as f:
            v=[int(x) for x in f.readline().split()[1:]]
        return sum(v),v[3]+(v[4] if len(v)>4 else 0)
    except (OSError,ValueError):
        return None

def ram_mib():
    try:
        values={}
        for line in Path("/proc/meminfo").read_text().splitlines():
            name,_,value=line.partition(":")
            values[name]=int(value.strip().split()[0])
        return (values["MemTotal"]-values["MemAvailable"])/1024
    except (OSError,ValueError,KeyError):
        return None

def gpu_state():
    if not shutil.which("nvidia-smi"):
        return []
    try:
        raw=subprocess.check_output([
            "nvidia-smi","--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits"
        ],text=True,stderr=subprocess.DEVNULL,timeout=3)
        gpu=[]
        for row in raw.splitlines():
            items=[x.strip() for x in row.split(",")]
            if len(items)!=6: continue
            def value(item):
                try: return float(item)
                except ValueError: return None
            gpu.append({"index":items[0],"name":items[1],
                        "vram_used_mib":value(items[2]),
                        "vram_total_mib":value(items[3]),
                        "gpu_util_percent":value(items[4]),
                        "gpu_power_watts":value(items[5])})
        return gpu
    except (OSError,subprocess.SubprocessError):
        return []

class Telemetry:
    def __init__(self,path,interval=0.5):
        self.path=Path(path)
        self.interval=interval
        self.stop=threading.Event()
        self.thread=None
        self.rows=[]
        self.summary={}
        self.gpu_description=None

    def _loop(self):
        last=cpu_ticks()
        with self.path.open("x",encoding="utf-8") as output:
            while not self.stop.is_set():
                now=cpu_ticks()
                busy=None
                if last is not None and now is not None:
                    total=now[0]-last[0]
                    if total>0: busy=100*(1-(now[1]-last[1])/total)
                last=now
                gpu=gpu_state()
                if gpu and self.gpu_description is None:
                    self.gpu_description=[{"index":x["index"],"name":x["name"],
                                          "vram_total_mib":x["vram_total_mib"]} for x in gpu]
                entry={"utc":datetime.now(timezone.utc).isoformat(),
                       "cpu_util_percent":busy,"host_ram_used_mib":ram_mib(),"gpus":gpu}
                output.write(json.dumps(entry,allow_nan=False)+"\n")
                output.flush()
                self.rows.append(entry)
                self.stop.wait(self.interval)

    def __enter__(self):
        self.thread=threading.Thread(target=self._loop,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set()
        self.thread.join(timeout=7)
        def measures(key):
            if key in ("cpu_util_percent","host_ram_used_mib"):
                return [x[key] for x in self.rows if x[key] is not None]
            return [g[key] for x in self.rows for g in x["gpus"] if g.get(key) is not None]
        self.summary={"samples":len(self.rows),"sampling_interval_seconds":self.interval,
                      "scope":"host-wide_not_attributable_to_single_model"}
        for key in ("cpu_util_percent","host_ram_used_mib",
                    "vram_used_mib","gpu_util_percent","gpu_power_watts"):
            sample=measures(key)
            self.summary[key]={"mean":statistics.mean(sample),"peak":max(sample)} if sample else None
        return False
