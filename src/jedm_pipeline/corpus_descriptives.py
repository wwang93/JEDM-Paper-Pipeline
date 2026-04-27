from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .io_utils import iter_txt_files, read_text


def _tokenize_alpha(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def _episode_stats(file_path: Path, stage: str) -> dict[str, int | str | float]:
    text = read_text(file_path)
    words = text.split()
    tokens = _tokenize_alpha(text)
    vocab = set(tokens)
    return {
        "stage": stage,
        "episode": file_path.name,
        "char_count": len(text),
        "word_count": len(words),
        "token_count": len(tokens),
        "vocab_size": len(vocab),
        "ttr": (len(vocab) / len(tokens)) if tokens else 0.0,
    }


def build_episode_descriptives(stage_dirs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, int | str | float]] = []
    for stage, directory in stage_dirs.items():
        for path in iter_txt_files(directory):
            rows.append(_episode_stats(path, stage=stage))
    if not rows:
        raise ValueError("No transcript files found in provided directories.")
    return pd.DataFrame(rows)


def build_stage_summary(episode_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["char_count", "word_count", "token_count", "vocab_size", "ttr"]
    summary = episode_df.groupby("stage")[metrics].agg(["mean", "std", "median", "min", "max"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()

    baseline = summary[summary["stage"] == "ASR"]
    if not baseline.empty:
        base = baseline.iloc[0]
        for metric in ["word_count_mean", "token_count_mean", "vocab_size_mean"]:
            col = f"delta_vs_ASR_{metric}_pct"
            summary[col] = (summary[metric] - base[metric]) / base[metric] * 100.0
    return summary


def save_descriptives(episode_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_df.sort_values(["stage", "episode"]).to_csv(out_dir / "episode_level_descriptives.csv", index=False)
    summary_df.to_csv(out_dir / "stage_summary_descriptives.csv", index=False)
