from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import setup_src_path

setup_src_path()

from jedm_pipeline.text_cleaning import normalize_string


def evaluate_reagan_case(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold_dict = {
        "PERSON": ["Ronald Reagan", "Jimmy Carter", "George HW Bush", "Mikhail Gorbachev", "Nancy Reagan", "John Poindexter", "Oliver North"],
        "ORG": ["Congress", "Moral Majority", "Supreme Court", "NATO", "Reagan administration", "Sandinista government", "Crash Course"],
        "GPE": ["America", "US", "Soviet Union", "Illinois", "New York", "Western Europe", "France", "Lebanon", "Nicaragua"],
        "LOC": ["Iron Curtain"],
        "NORP": ["American", "African Americans", "conservatives", "religious conservatives", "economic conservatives", "Cold War hawks", "Christian right", "anti-government crusaders", "Democratic", "Soviet", "Iranian", "Middle Eastern"],
        "EVENT": ["Reagan Revolution", "Cold War", "Iran-Contra Scandal", "New Deal", "Great Society", "Korean", "Vietnam wars", "1960s", "1970s", "1980s", "mid-1990s", "atomic age", "FREEZE movement"],
        "LAW": ["Economic Bill of Rights", "Tax Reform Act", "Anti-ballistic Missile Treaty"],
    }

    system_files = {
        "ASR": "ASRNER.csv",
        "LLMsCoref": "LLMscoreNER.csv",
        "LLMsNER": "LLMs_NER.csv",
    }

    summary_rows = []
    detail_rows = []
    for system, filename in system_files.items():
        df = pd.read_csv(base_dir / filename)
        df["entity_norm"] = df["entity"].apply(normalize_string)
        df["ner_type_norm"] = df["ner_type"].apply(normalize_string)
        for ner_type, entities in gold_dict.items():
            pred_set = set(df[df["ner_type_norm"] == ner_type.lower()]["entity_norm"])
            gold_set = {normalize_string(e) for e in entities}
            tp = len(pred_set & gold_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0
            summary_rows.append(
                {
                    "system": system,
                    "ner_type": ner_type,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                }
            )
            for entity in entities:
                detail_rows.append(
                    {
                        "system": system,
                        "ner_type": ner_type,
                        "entity": entity,
                        "recognized": "Y" if normalize_string(entity) in pred_set else "N",
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    summary_df, detail_df = evaluate_reagan_case(base_dir)
    summary_df.to_csv(base_dir / "ner_strict_eval_summary.csv", index=False)
    detail_df.to_csv(base_dir / "ner_strict_eval_detail.csv", index=False)
    macro_df = summary_df.groupby("system")[["precision", "recall", "f1"]].mean().reset_index()
    print(macro_df.to_string(index=False))


if __name__ == "__main__":
    main()
