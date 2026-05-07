from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABELS = ["PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "LAW"]


def build_token_summary(df: pd.DataFrame) -> pd.DataFrame:
    by_ep = df[df["llm_label"].isin(LABELS)].groupby(["doc_id", "llm_label"]).size().unstack(fill_value=0)
    for label in LABELS:
        if label not in by_ep.columns:
            by_ep[label] = 0
    by_ep = by_ep[LABELS]

    rows = []
    for label in LABELS:
        s = by_ep[label]
        rows.append(
            {
                "entity_type": label,
                "total_count": int(s.sum()),
                "mean_per_episode": float(s.mean()),
                "sd_per_episode": float(s.std(ddof=1)),
                "episodes_with_ge1_mention": int((s >= 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_mention_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for doc_id, group in df.groupby("doc_id", sort=False):
        labels = group["llm_label"].astype(str).tolist()
        counts = {label: 0 for label in LABELS}
        prev = "none"
        for label in labels:
            if label in LABELS and label != prev:
                counts[label] += 1
            prev = label
        row = {"doc_id": doc_id}
        row.update(counts)
        rows.append(row)

    by_ep = pd.DataFrame(rows).set_index("doc_id")
    for label in LABELS:
        if label not in by_ep.columns:
            by_ep[label] = 0
    by_ep = by_ep[LABELS]

    summary_rows = []
    for label in LABELS:
        s = by_ep[label]
        summary_rows.append(
            {
                "entity_type": label,
                "total_count": int(s.sum()),
                "mean_per_episode": float(s.mean()),
                "sd_per_episode": float(s.std(ddof=1)),
                "episodes_with_ge1_mention": int((s >= 1).sum()),
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corpus-level entity profile summary tables for RQ2.")
    parser.add_argument(
        "--labels-csv",
        default="data/processed/full_corpus_llm55_alltokens_v1/all_docs_token_labels.csv",
        help="Token-level labels CSV from full corpus annotation run.",
    )
    parser.add_argument("--output-dir", default="data/processed/full_corpus_llm55_alltokens_v1")
    args = parser.parse_args()

    labels_path = Path(args.labels_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_path)
    token_summary = build_token_summary(df)
    mention_summary = build_mention_summary(df)

    token_summary.to_csv(out_dir / "rq2_entity_profile_token_summary.csv", index=False)
    mention_summary.to_csv(out_dir / "rq2_entity_profile_mention_summary.csv", index=False)

    print("Saved token summary:", out_dir / "rq2_entity_profile_token_summary.csv")
    print("Saved mention summary:", out_dir / "rq2_entity_profile_mention_summary.csv")


if __name__ == "__main__":
    main()
