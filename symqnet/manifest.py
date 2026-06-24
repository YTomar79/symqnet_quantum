from __future__ import annotations

from dataclasses import asdict, is_dataclass
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from .provenance import stable_config_hash


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _scrub_path_string(value: str) -> str:
    root = _project_root()
    home = Path.home()
    out = value.replace(str(root), "<PROJECT_ROOT>")
    out = out.replace(str(home), "<HOME>")
    return out


def anonymize_manifest(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: anonymize_manifest(item) for key, item in value.items()}
    if isinstance(value, list):
        return [anonymize_manifest(item) for item in value]
    if isinstance(value, tuple):
        return [anonymize_manifest(item) for item in value]
    if isinstance(value, str):
        if value == sys.executable or value.endswith("/.venv/bin/python"):
            return "python"
        return _scrub_path_string(value)
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(cwd: str | Path) -> dict[str, object]:
    cwd = Path(cwd)

    def run(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return proc.stdout.strip()

    root = run(["git", "rev-parse", "--show-toplevel"])
    if root is None:
        return {"is_git_repo": False}
    return {
        "is_git_repo": True,
        "root": root,
        "commit": run(["git", "rev-parse", "HEAD"]) or "",
        "branch": run(["git", "branch", "--show-current"]) or "",
        "status_short": run(["git", "status", "--short"]) or "",
    }


def package_versions(names: tuple[str, ...] = ("numpy", "torch", "matplotlib", "scipy", "pytest")) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = ""
    return versions


def task_bank_metadata(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    task_path = Path(path)
    if not task_path.exists():
        return {"path": str(task_path), "exists": False}
    out: dict[str, object] = {"path": str(task_path), "exists": True, "sha256": file_sha256(task_path)}
    try:
        with np.load(task_path) as data:
            out.update(
                {
                    "count": int(data["J"].shape[0]),
                    "n_qubits": int(data["n_qubits"]) if "n_qubits" in data else int(data["h"].shape[1]),
                    "seed": int(data["seed"]) if "seed" in data else "",
                    "j_range": data["j_range"].astype(float).tolist() if "j_range" in data else "",
                    "h_range": data["h_range"].astype(float).tolist() if "h_range" in data else "",
                }
            )
    except Exception as exc:  # pragma: no cover - defensive manifest metadata only
        out["error"] = str(exc)
    return out


def vae_checkpoint_metadata(config: object) -> dict[str, object]:
    model = getattr(config, "model", None)
    if model is None or not getattr(model, "use_vae", False):
        return {"used": False}
    checkpoint = Path(getattr(model, "vae_checkpoint", ""))
    if not checkpoint.exists():
        return {"used": True, "path": str(checkpoint), "exists": False}
    out: dict[str, object] = {"used": True, "path": str(checkpoint), "exists": True, "sha256": file_sha256(checkpoint)}
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("pretrain_metadata", {}) if isinstance(payload, dict) else {}
        out["pretrain_metadata"] = metadata
    except Exception as exc:  # pragma: no cover - manifest metadata should not stop a completed run.
        out["error"] = str(exc)
    return out


def csv_shape(path: str | Path) -> dict[str, object]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {"path": str(csv_path), "exists": False}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = sum(1 for _ in reader)
        return {"path": str(csv_path), "exists": True, "rows": rows, "columns": reader.fieldnames or []}


def build_manifest(
    *,
    run_root: str | Path,
    config: object,
    args: dict[str, object],
    commands: list[list[str]],
    task_bank: str | Path | None,
    started_at: float,
    files: list[str | Path] | None = None,
    outputs: list[str | Path] | None = None,
    anonymize: bool = False,
) -> dict[str, object]:
    files = files or []
    outputs = outputs or []
    file_hashes = {}
    for path in files:
        p = Path(path)
        if p.exists() and p.is_file():
            file_hashes[str(p)] = file_sha256(p)
    payload = asdict(config) if is_dataclass(config) else config
    manifest = {
        "run_root": str(run_root),
        "created_unix": time.time(),
        "elapsed_sec": time.time() - started_at,
        "python": sys.version,
        "platform": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
            "cpu_count": os.cpu_count(),
        },
        "torch": {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        },
        "packages": package_versions(),
        "git": git_state(Path.cwd()),
        "config_hash": stable_config_hash(config),
        "config": payload,
        "args": args,
        "commands": commands,
        "task_bank": task_bank_metadata(task_bank),
        "vae_checkpoint": vae_checkpoint_metadata(config),
        "file_hashes": file_hashes,
        "outputs": [csv_shape(path) if str(path).endswith(".csv") else {"path": str(path), "exists": Path(path).exists()} for path in outputs],
    }
    return anonymize_manifest(manifest) if anonymize else manifest


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
