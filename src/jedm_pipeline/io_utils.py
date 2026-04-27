from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def iter_txt_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.txt"))


def iter_srt_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.srt"))
