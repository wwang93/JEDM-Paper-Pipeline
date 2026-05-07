from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

import pandas as pd

from jedm_pipeline.config import load_config
from jedm_pipeline.io_utils import iter_txt_files, read_text
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


def tokenize_text(text: str) -> list[str]:
    return text.split()


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
        if not left.isdigit():
            continue
        preds[int(left)] = normalize_label(right.strip())
    return preds


def chunk_rows(rows: list[dict], batch_size: int):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run token-level LLM annotation for full ASR corpus.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--input-dir",
        default="data/external/paper_meta_data/Paper Meta Data/USHistory_raw_txts",
    )
    parser.add_argument(
        "--codebook-xlsx",
        default="data/Human_The_Reagan_Revolution_Crash_Course_US_History_43_en_one_column_words.xlsx",
    )
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--output-dir", default="data/processed/full_corpus_llm55_alltokens_v1")
    parser.add_argument("--resume", action="store_true", help="Skip documents with existing per-doc output")
    args = parser.parse_args()

    config = load_config(args.config)
    enhancer = OpenAITextEnhancer(
        api_key=config["env"]["OPENAI_API_KEY"],
        model=args.model,
        max_chars=120000,
        sleep_seconds=config["llm"]["sleep_seconds"],
        max_retries=config["llm"]["max_retries"],
    )

    codebook_text = load_codebook_text(Path(args.codebook_xlsx))
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    per_doc_dir = out_dir / "per_doc"
    raw_dir = out_dir / "raw_outputs"
    per_doc_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for txt_path in iter_txt_files(input_dir):
        per_doc_path = per_doc_dir / f"{txt_path.stem}.token_labels.csv"
        raw_out_path = raw_dir / f"{txt_path.stem}.raw_outputs.csv"
        if args.resume and per_doc_path.exists() and raw_out_path.exists():
            doc_df = pd.read_csv(per_doc_path)
            for _, rr in doc_df.iterrows():
                all_rows.append(
                    {
                        "doc_id": rr["doc_id"],
                        "token_id": int(rr["token_id"]),
                        "word": rr["word"],
                        "llm_label": normalize_label(rr["llm_label"]),
                    }
                )
            pred_counts = doc_df["llm_label"].value_counts().to_dict()
            manifest_rows.append(
                {
                    "doc_id": txt_path.name,
                    "n_tokens": int(len(doc_df)),
                    "n_predicted_entity_tokens": int((doc_df["llm_label"] != "none").sum()),
                    "label_counts": str(pred_counts),
                }
            )
            print(f"Skipped {txt_path.name} (existing output)")
            continue

        text = read_text(txt_path)
        tokens = tokenize_text(text)
        token_rows = [{"token_id": i + 1, "word": w} for i, w in enumerate(tokens)]
        predictions: dict[int, str] = {}
        batch_logs: list[dict] = []

        for b_idx, batch in enumerate(chunk_rows(token_rows, args.batch_size), start=1):
            prompt = build_prompt(codebook_text, batch)
            output = enhancer._chat(
                system_prompt="You are a strict annotation assistant.",
                user_prompt=prompt,
            )
            batch_logs.append({"batch": b_idx, "start_token_id": batch[0]["token_id"], "output": output})
            parsed = parse_csv_like(output)
            predictions.update(parsed)

        doc_rows = []
        for r in token_rows:
            label = normalize_label(predictions.get(r["token_id"], "none"))
            row = {
                "doc_id": txt_path.name,
                "token_id": r["token_id"],
                "word": r["word"],
                "llm_label": label,
            }
            doc_rows.append(row)
            all_rows.append(row)

        doc_df = pd.DataFrame(doc_rows)
        doc_df.to_csv(per_doc_path, index=False)

        with raw_out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["batch", "start_token_id", "output"])
            writer.writeheader()
            writer.writerows(batch_logs)

        pred_counts = doc_df["llm_label"].value_counts().to_dict()
        manifest_rows.append(
            {
                "doc_id": txt_path.name,
                "n_tokens": len(token_rows),
                "n_predicted_entity_tokens": int((doc_df["llm_label"] != "none").sum()),
                "label_counts": str(pred_counts),
            }
        )
        print(f"Annotated {txt_path.name}: tokens={len(token_rows)}")

    pd.DataFrame(all_rows).to_csv(out_dir / "all_docs_token_labels.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(out_dir / "annotation_manifest.csv", index=False)

    print(f"Done. docs={len(manifest_rows)}, total_tokens={len(all_rows)}")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
