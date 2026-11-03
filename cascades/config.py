"""Load versioned CascadeOne cascade YAML definitions.

A cascade definition is distinct from a benchmark setup: the YAML describes
the model pair, backend identity and routing policy; benchmark.json describes
a dataset, adapter and selection for a particular experiment.
"""
from __future__ import annotations

from pathlib import Path
import yaml


SCHEMA = "cascadeone-cascade-v0.1"


def load_cascade(path: str | Path) -> dict:
    """Parse one cascade definition with PyYAML's non-executing safe loader."""
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError(f"invalid cascade definition/schema: {path}")
    if not isinstance(config.get("id"), str) or not config["id"]:
        raise ValueError(f"cascade ID is required: {path}")
    for role in ("fast", "strong"):
        model = config.get(role)
        if not isinstance(model, dict) or not isinstance(model.get("model_id"), str) or not model["model_id"]:
            raise ValueError(f"cascade {role}.model_id is required: {path}")
    routing = config.get("routing")
    if not isinstance(routing, dict) or not isinstance(routing.get("type"), str) or not routing["type"]:
        raise ValueError(f"cascade routing.type is required: {path}")
    return config


def list_cascades(root: str | Path):
    """Yield documented, launchable reference cascade profiles."""
    root = Path(root)
    for path in sorted((root / "cascades").glob("*/cascade.yaml")):
        if (path.parent / "compose.yaml").is_file():
            yield path.parent.name, load_cascade(path)
