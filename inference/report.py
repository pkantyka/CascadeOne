"""Display validated paired benchmark evidence without inventing performance claims."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from tools.validate_result import validate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("missing or invalid benchmark metric: " + name)
    return float(value)


def read_report(cascade_path, config, run_path, *, min_speedup_pct=10.0,
                max_quality_loss_pp=2.0, max_p95_ratio=1.10):
    """Compare this *exact* cascade profile to strong-only under identical residency.

    Invalid/incompatible evidence is rejected, not silently displayed as a success.
    Thresholds are user-controlled alert levels, not statistical significance tests.
    """
    if any(not math.isfinite(v) or v < 0 for v in
           (min_speedup_pct, max_quality_loss_pp, max_p95_ratio)):
        raise ValueError("alert thresholds must be finite non-negative numbers")
    if max_p95_ratio <= 0:
        raise ValueError("max_p95_ratio must be positive")
    root = Path(run_path)
    validate(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if manifest["phase"] != "colocated":
        raise ValueError("a colocated benchmark is required for a cascade/strong comparison")
    if manifest.get("execution") != "systemone_inference_proxy_http":
        raise ValueError("benchmark uses the legacy execution path; rerun via the shared inference API")
    if manifest["cascade_id"] != config["id"] or manifest["cascade_config_sha256"] != sha(cascade_path):
        raise ValueError("benchmark does not match the exact cascade configuration SHA-256")
    if manifest["routing"] != config["routing"] or manifest["models"] != {
        "fast": config["fast"], "strong": config["strong"]
    }:
        raise ValueError("benchmark model/policy mismatch")
    arms = summary["arms"]
    if set(arms) != {"fast_only", "strong_only", "cascade"}:
        raise ValueError("benchmark must contain all three paired arms")
    strong, casc = arms["strong_only"], arms["cascade"]
    if strong["n"] != casc["n"] or strong["unique_requests"] != casc["unique_requests"]:
        raise ValueError("benchmark populations do not match")
    trials = [json.loads(x) for x in (root / "measurements.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {arm: {(x["id"], x["repeat"]) for x in trials if x["arm"] == arm} for arm in arms}
    if keys["strong_only"] != keys["cascade"] or keys["strong_only"] != keys["fast_only"]:
        raise ValueError("benchmark arms are not paired on identical requests")
    s_mean = finite(strong["mean_ms"], "strong.mean_ms")
    c_mean = finite(casc["mean_ms"], "cascade.mean_ms")
    s_p95 = finite(strong["p95_ms"], "strong.p95_ms")
    c_p95 = finite(casc["p95_ms"], "cascade.p95_ms")
    if min(s_mean, c_mean, s_p95, c_p95) <= 0:
        raise ValueError("non-positive benchmark latency")
    si_mean = finite(strong["engine_mean_ms"], "strong.engine_mean_ms")
    ci_mean = finite(casc["engine_mean_ms"], "cascade.engine_mean_ms")
    si_p95 = finite(strong["engine_p95_ms"], "strong.engine_p95_ms")
    ci_p95 = finite(casc["engine_p95_ms"], "cascade.engine_p95_ms")
    if min(si_mean, ci_mean, si_p95, ci_p95) <= 0:
        raise ValueError("non-positive internal engine latency")
    engine_saving = 100 * (1 - ci_mean / si_mean)
    saving = 100 * (1 - c_mean / s_mean)
    factor = s_mean / c_mean
    tail = c_p95 / s_p95
    sq, cq = strong.get("question_expected_accuracy"), casc.get("question_expected_accuracy")
    warnings = []
    if (sq is None) != (cq is None):
        raise ValueError("partial Gold comparison between arms")
    quality_delta = None
    if sq is not None:
        sq, cq = finite(sq, "strong.quality"), finite(cq, "cascade.quality")
        if not (0 <= sq <= 1 and 0 <= cq <= 1):
            raise ValueError("quality must be within [0,1]")
        quality_delta = 100 * (cq - sq)
        if quality_delta < -max_quality_loss_pp - 1e-9:
            warnings.append(f"Quality loss {abs(quality_delta):.2f} pp exceeds {max_quality_loss_pp:.2f} pp alert limit.")
    else:
        warnings.append("Gold quality unavailable: cannot assess cascade quality loss.")
    if saving < min_speedup_pct:
        warnings.append(f"Mean latency reduction {saving:+.1f}% is below the {min_speedup_pct:.1f}% minimum.")
    if tail > max_p95_ratio:
        warnings.append(f"p95 latency ratio {tail:.2f} exceeds the {max_p95_ratio:.2f} alert limit.")
    dataset_id = manifest.get("dataset_identity", {}).get("id") or "unregistered dataset"
    setup = manifest.get("benchmark_setup", {})
    setup_id = setup.get("id")
    if "smoke" in str(dataset_id).lower() or "smoke" in str(setup_id).lower():
        warnings.append("Synthetic smoke benchmark is only an integration check, not performance or quality evidence.")
    if "train" in str(dataset_id).lower() or manifest.get("dataset_identity", {}).get("split") == "train":
        warnings.append("Training-source measurements are not independent held-out evidence.")
    if manifest["n_requests"] < 30:
        warnings.append(f"Only {manifest['n_requests']} unique requests: estimates may be unstable.")
    warnings.append("Historical sequential HTTP proxy benchmark; workload, hardware or concurrency may differ from the current service.")
    return {
        "benchmark": str(root),
        "dataset_id": dataset_id,
        "setup_id": setup_id,
        "n_requests": manifest["n_requests"],
        "mean_latency_ms": {"cascade": c_mean, "strong": s_mean},
        "engine_mean_latency_ms": {"cascade": ci_mean, "strong": si_mean},
        "engine_p95_latency_ms": {"cascade": ci_p95, "strong": si_p95},
        "engine_latency_reduction_pct": engine_saving,
        "p95_latency_ms": {"cascade": c_p95, "strong": s_p95},
        "latency_reduction_pct": saving,
        "speedup_factor": factor,
        "p95_ratio": tail,
        "quality": None if quality_delta is None else {
            "cascade": cq, "strong": sq, "delta_pp": quality_delta
        },
        "alert_thresholds": {
            "min_speedup_pct": min_speedup_pct,
            "max_quality_loss_pp": max_quality_loss_pp,
            "max_p95_ratio": max_p95_ratio
        },
        "warnings": warnings,
    }


def startup_lines(report):
    if report is None:
        return [
            "[CascadeOne][WARN] No benchmark run provided: speedup and quality difference vs strong are UNKNOWN.",
            "[CascadeOne][WARN] Pass --benchmark RUN_DIR (or CASCADEONE_BENCHMARK_RUN) for validated historical evidence."
        ]
    lines = [
        f"[CascadeOne] Benchmark: {report['benchmark']} ({report['dataset_id']}, {report['n_requests']} requests)",
        f"[CascadeOne] End-to-end mean latency: cascade {report['mean_latency_ms']['cascade']:.1f} ms, "
        f"strong {report['mean_latency_ms']['strong']:.1f} ms; "
        f"reduction {report['latency_reduction_pct']:+.1f}%, "
        f"speedup {report['speedup_factor']:.2f}x.",
        f"[CascadeOne] Engine-internal mean: cascade {report['engine_mean_latency_ms']['cascade']:.1f} ms, "
        f"strong {report['engine_mean_latency_ms']['strong']:.1f} ms; "
        f"reduction {report['engine_latency_reduction_pct']:+.1f}%.",
        f"[CascadeOne] p95: cascade {report['p95_latency_ms']['cascade']:.1f} ms, "
        f"strong {report['p95_latency_ms']['strong']:.1f} ms; "
        f"ratio {report['p95_ratio']:.2f}."
    ]
    quality = report["quality"]
    if quality is not None:
        lines.append(f"[CascadeOne] Gold-based question accuracy: cascade {quality['cascade'] * 100:.2f}%, "
                     f"strong {quality['strong'] * 100:.2f}%; "
                     f"delta {quality['delta_pp']:+.2f} pp.")
    for warning in report["warnings"]:
        lines.append("[CascadeOne][WARN] " + warning)
    return lines
