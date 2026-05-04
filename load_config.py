"""Load experiment hyperparameters from JSON (no defaults in algorithm code)."""
from __future__ import annotations

import json
from types import SimpleNamespace


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(x) for x in obj]
    return obj


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError("config root must be a JSON object")
    return _to_ns(data)
