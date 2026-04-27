from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase, cer, mer, wer, wil

from .io_utils import ensure_dir, read_text


def evaluate_asr_files(file_map: dict[str, Path], human_key: str, compare_keys: list[str]) -> pd.DataFrame:
    texts: dict[str, str] = {k: read_text(v) for k, v in file_map.items()}
    transform = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
    truth = transform(texts[human_key])

    rows = []
    for key in compare_keys:
        hypo = transform(texts[key])
        rows.append(
            {
                "model": key,
                "WER": wer(truth, hypo),
                "MER": mer(truth, hypo),
                "WIL": wil(truth, hypo),
                "CER": cer(truth, hypo),
            }
        )
    return pd.DataFrame(rows).sort_values("WER")


def save_asr_results(df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    csv_path = out_dir / "rq1_asr_metrics.csv"
    fig_path = out_dir / "rq1_asr_metrics.png"
    df.to_csv(csv_path, index=False)

    plot_df = df.set_index("model")
    ax = plot_df[["WER", "MER", "WIL", "CER"]].plot(kind="bar", figsize=(10, 6), rot=0)
    ax.set_title("ASR Metrics vs Human Transcript")
    ax.set_ylabel("Error / proportion")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
