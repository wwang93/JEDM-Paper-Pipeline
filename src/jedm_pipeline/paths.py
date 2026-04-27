from __future__ import annotations

from pathlib import Path


def resolve_path(config: dict, key: str) -> Path:
    root = Path(config["_project_root"])
    rel = config["paths"][key]
    path = Path(rel)
    if path.is_absolute():
        return path
    return root / path
