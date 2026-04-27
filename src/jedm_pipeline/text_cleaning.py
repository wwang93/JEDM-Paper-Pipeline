from __future__ import annotations

import re
import textwrap
from pathlib import Path

from .io_utils import ensure_dir, iter_srt_files, read_text, write_text


def srt_to_text(srt_content: str) -> str:
    lines = srt_content.splitlines()
    text_lines: list[str] = []
    for line in lines:
        if re.match(r"^\d+\s*$", line):
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2},\d{3}", line):
            continue
        if not line.strip():
            continue
        text_lines.append(line.strip())
    return " ".join(text_lines)


def convert_srt_file(srt_path: Path, txt_path: Path) -> None:
    content = read_text(srt_path)
    converted = srt_to_text(content)
    write_text(txt_path, converted)


def batch_convert_srt_dir(srt_dir: Path, txt_dir: Path) -> int:
    ensure_dir(txt_dir)
    count = 0
    for srt_path in iter_srt_files(srt_dir):
        txt_path = txt_dir / f"{srt_path.stem}.txt"
        convert_srt_file(srt_path, txt_path)
        count += 1
    return count


def normalize_string(value: str) -> str:
    return str(value).strip().lower()


def chunk_paragraphs(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        if not para.strip():
            continue
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            chunks.extend(textwrap.wrap(para, max_chars, break_long_words=False))
    return chunks
