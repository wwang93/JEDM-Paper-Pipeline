from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from _bootstrap import setup_src_path

setup_src_path()

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from jedm_pipeline.config import load_config
from jedm_pipeline.llm_client import OpenAITextEnhancer


ALLOWED_LABELS = ["PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "LAW", "none"]


def normalize_label(value: str) -> str:
    raw = str(value).strip()
    mapping = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "ORG": "ORG",
        "GPE": "GPE",
        "LOC": "LOC",
        "NORP": "NORP",
        "EVENT": "EVENT",
        "LAW": "LAW",
        "NONE": "none",
        "none": "none",
        "": "none",
    }
    key = raw.upper() if raw not in {"none", ""} else raw
    return mapping.get(key, "none")


def load_codebook_text(workbook_path: Path) -> str:
    sheet = pd.read_excel(workbook_path, sheet_name="codebook", header=None)
    rows = []
    for _, row in sheet.iterrows():
        values = [str(v).strip() for v in row.tolist() if pd.notna(v) and str(v).strip()]
        if values:
            rows.append(" | ".join(values))
    return "\n".join(rows)


def batched(items: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def build_prompt(codebook_text: str, token_rows: list[dict]) -> str:
    header = (
        "You are doing token-level NER coding using a fixed codebook.\n"
        "Use ONLY these labels: PERSON, ORG, GPE, LOC, NORP, EVENT, LAW, none.\n"
        "If a token is not a named entity by the codebook, label it as none.\n"
        "Return ONLY CSV lines with two columns: token_id,label\n"
        "No explanations, no extra text.\n\n"
        "Codebook:\n"
        f"{codebook_text}\n\n"
        "Tokens to code:\n"
    )
    lines = [f"{r['token_id']},{r['word']}" for r in token_rows]
    return header + "\n".join(lines)


def parse_csv_like(output: str) -> dict[int, str]:
    preds: dict[int, str] = {}
    for line in output.splitlines():
        s = line.strip()
        if not s or "," not in s:
            continue
        left, right = s.split(",", 1)
        left = left.strip()
        right = normalize_label(right.strip())
        if not left.isdigit():
            continue
        preds[int(left)] = right
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM token-level coding on first 600 Reagan tokens.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--compare-csv",
        default="data/processed/reagan_consensus_prep/reagan_compare_cleaned.csv",
    )
    parser.add_argument(
        "--codebook-xlsx",
        default="data/Human_The_Reagan_Revolution_Crash_Course_US_History_43_en_one_column_words.xlsx",
    )
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--output-dir", default="data/processed/reagan_llm_eval")
    parser.add_argument("--model", default=None, help="Optional model override, e.g., gpt-5.5")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = args.model if args.model else config["llm"]["model"]
    enhancer = OpenAITextEnhancer(
        api_key=config["env"]["OPENAI_API_KEY"],
        model=model_name,
        max_chars=100000,
        sleep_seconds=config["llm"]["sleep_seconds"],
        max_retries=config["llm"]["max_retries"],
    )

    compare_df = pd.read_csv(args.compare_csv)
    subset = compare_df.head(args.n).copy()
    subset["consensus_label"] = subset["consensus_label"].apply(normalize_label)
    token_rows = subset[["token_id", "word", "consensus_label"]].to_dict(orient="records")

    codebook_text = load_codebook_text(Path(args.codebook_xlsx))
    predictions: dict[int, str] = {}
    raw_outputs: list[dict[str, str | int]] = []

    for idx, chunk in enumerate(batched(token_rows, args.batch_size), start=1):
        prompt = build_prompt(codebook_text, chunk)
        output = enhancer._chat(
            system_prompt="You are a strict annotation assistant.",
            user_prompt=prompt,
        )
        raw_outputs.append({"batch": idx, "output": output})
        parsed = parse_csv_like(output)
        predictions.update(parsed)

    pred_labels = []
    true_labels = []
    rows = []
    for r in token_rows:
        tid = int(r["token_id"])
        gold = normalize_label(str(r["consensus_label"]))
        pred = normalize_label(predictions.get(tid, "none"))
        true_labels.append(gold)
        pred_labels.append(pred)
        rows.append({"token_id": tid, "word": r["word"], "gold": gold, "pred": pred})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(out_dir / "reagan_600_gold_vs_llm.csv", index=False)
    pd.DataFrame(raw_outputs).to_csv(out_dir / "reagan_600_llm_raw_outputs.csv", index=False)

    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, labels=ALLOWED_LABELS, average="macro", zero_division=0)
    micro_f1 = f1_score(true_labels, pred_labels, labels=ALLOWED_LABELS, average="micro", zero_division=0)

    summary = {
        "n_tokens": len(true_labels),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "model": model_name,
    }
    with (out_dir / "reagan_600_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report = classification_report(
        true_labels,
        pred_labels,
        labels=ALLOWED_LABELS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(out_dir / "reagan_600_classification_report.csv", index=True)

    cm = confusion_matrix(true_labels, pred_labels, labels=ALLOWED_LABELS)
    with (out_dir / "reagan_600_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred"] + ALLOWED_LABELS)
        for label, row in zip(ALLOWED_LABELS, cm.tolist()):
            writer.writerow([label] + row)

    print(f"Done. n={len(true_labels)} accuracy={acc:.4f} macro_f1={macro_f1:.4f} micro_f1={micro_f1:.4f}")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
