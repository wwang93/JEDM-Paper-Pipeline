from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir
from .text_cleaning import normalize_string


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["entity_norm"] = out["entity"].apply(normalize_string)
    out["ner_type_norm"] = out["ner_type"].apply(normalize_string)
    return out


def entity_level_metrics(pred_df: pd.DataFrame, gold_df: pd.DataFrame, model_name: str) -> dict:
    pred = _normalize_df(pred_df)
    gold = _normalize_df(gold_df)
    pred_set = set(zip(pred["entity_norm"], pred["ner_type_norm"]))
    gold_set = set(zip(gold["entity_norm"], gold["ner_type_norm"]))

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    acc = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "model": model_name,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compare_ner_systems(system_csvs: dict[str, Path], gold_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold_df = pd.read_csv(gold_csv)
    summary = []
    by_type_rows = []

    system_data = {name: pd.read_csv(path) for name, path in system_csvs.items()}
    for name, pred_df in system_data.items():
        summary.append(entity_level_metrics(pred_df, gold_df, name))

    all_types = set(_normalize_df(gold_df)["ner_type_norm"])
    for df in system_data.values():
        all_types |= set(_normalize_df(df)["ner_type_norm"])

    for ner_type in sorted(all_types):
        gold_sub = _normalize_df(gold_df)
        gold_sub = gold_sub[gold_sub["ner_type_norm"] == ner_type]
        row: dict[str, float | str] = {"ner_type": ner_type.upper()}
        for name, pred_df in system_data.items():
            pred_sub = _normalize_df(pred_df)
            pred_sub = pred_sub[pred_sub["ner_type_norm"] == ner_type]
            metric = entity_level_metrics(pred_sub, gold_sub, name)
            row[f"{name}_f1"] = metric["f1"]
            row[f"{name}_precision"] = metric["precision"]
            row[f"{name}_recall"] = metric["recall"]
            row[f"{name}_accuracy"] = metric["accuracy"]
        by_type_rows.append(row)

    return pd.DataFrame(summary), pd.DataFrame(by_type_rows)


def save_ner_results(summary_df: pd.DataFrame, by_type_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    summary_df.to_csv(out_dir / "rq2_ner_summary.csv", index=False)
    by_type_df.to_csv(out_dir / "rq2_ner_by_type.csv", index=False)
