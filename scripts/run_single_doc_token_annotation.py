from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

import pandas as pd

from jedm_pipeline.config import load_config
from jedm_pipeline.io_utils import read_text
from jedm_pipeline.llm_client import OpenAITextEnhancer


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
    return header + "\n".join(f"{r['token_id']},{r['word']}" for r in token_rows)


def parse_csv_like(output: str) -> dict[int, str]:
    preds: dict[int, str] = {}
    for line in output.splitlines():
        s = line.strip()
        if not s or "," not in s:
            continue
        left, right = s.split(",", 1)
        left = left.strip()
        if not left.isdigit():
            continue
        preds[int(left)] = normalize_label(right.strip())
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Token-level LLM annotation for one transcript file.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--codebook-xlsx", default="data/Human_The_Reagan_Revolution_Crash_Course_US_History_43_en_one_column_words.xlsx")
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--start-token", type=int, default=1, help="1-based start token index in input file")
    parser.add_argument("--end-token", type=int, default=None, help="1-based end token index in input file (inclusive)")
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    enhancer = OpenAITextEnhancer(
        api_key=config["env"]["OPENAI_API_KEY"],
        model=args.model,
        max_chars=120000,
        sleep_seconds=config["llm"]["sleep_seconds"],
        max_retries=config["llm"]["max_retries"],
    )

    all_tokens = read_text(Path(args.input_file)).split()
    start_idx = max(args.start_token - 1, 0)
    if args.end_token is not None:
        end_idx = min(args.end_token, len(all_tokens))
    else:
        end_idx = min(start_idx + args.n, len(all_tokens))
    tokens = all_tokens[start_idx:end_idx]
    rows = [{"token_id": i + 1, "word": t} for i, t in enumerate(tokens)]
    codebook_text = load_codebook_text(Path(args.codebook_xlsx))

    preds: dict[int, str] = {}
    raw_logs: list[dict[str, str | int]] = []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        prompt = build_prompt(codebook_text, chunk)
        output = enhancer._chat("You are a strict annotation assistant.", prompt)
        raw_logs.append({"batch": (i // args.batch_size) + 1, "start_token_id": chunk[0]["token_id"], "output": output})
        preds.update(parse_csv_like(output))

    out_rows = []
    for r in rows:
        out_rows.append({"token_id": r["token_id"], "word": r["word"], "llm_label": normalize_label(preds.get(r["token_id"], "none"))})

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)

    raw_path = out_path.with_suffix(".raw_outputs.csv")
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["batch", "start_token_id", "output"])
        writer.writeheader()
        writer.writerows(raw_logs)

    print(f"Saved token labels: {out_path}")
    print(f"Source token window: start={start_idx+1}, end={start_idx+len(tokens)}, n={len(tokens)}")


if __name__ == "__main__":
    main()
