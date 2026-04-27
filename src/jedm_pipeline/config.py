from __future__ import annotations

from pathlib import Path
from typing import Any

import os

import yaml
from dotenv import load_dotenv


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    root = project_root()
    load_dotenv(root / ".env")
    load_dotenv(root / ".env.example", override=False)
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config["_project_root"] = str(root)
    config.setdefault("env", {})
    config["env"]["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    return config
