from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import platform

import torch


def stable_config_hash(cfg: object) -> str:
    """Short stable hash for the effective experiment configuration."""
    payload = asdict(cfg) if is_dataclass(cfg) else cfg
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def hardware_label(device: torch.device | str) -> str:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(device.index or 0)}"
    if device.type == "mps":
        return "mps:apple-silicon"
    return f"cpu:{platform.machine() or 'unknown'}"
