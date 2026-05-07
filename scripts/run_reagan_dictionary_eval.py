from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GOLD_DICT = {
    "PERSON": [
        "Ronald Reagan",
        "Jimmy Carter",
        "George HW Bush",
        "Mikhail Gorbachev",
        "Nancy Reagan",
        "John Poindexter",
        "Oliver North",
    ],
    "ORG": [
        "Congress",
        "Moral Majority",
        "Supreme Court",
        "NATO",
        "Reagan administration",
        "Sandinista government",
        "Crash Course",
    ],
    "GPE": [
        "America",
        "US",
        "Soviet Union",
        "Illinois",
        "New York",
        "Western Europe",
        "France",
        "Lebanon",
        "Nicaragua",
    ],
    "LOC": ["Iron Curtain"],
    "NORP": [
        "American",
        "African Americans",
        "conservatives",
        "religious conservatives",
        "economic conservatives",
        "Cold War hawks",
        "Christian right",
        "anti-government crusaders",
        "Democratic",
        "Soviet",
        "Iranian",
        "Middle Eastern",
    ],
    "EVENT": [
        "Reagan Revolution",
        "Cold War",
        "Iran-Contra Scandal",
        "New Deal",
        "Great Society",
        "Korean",
        "Vietnam War",
        "1960s",
        "1970s",
        "1980s",
        "mid-1990s",
        "atomic age",
        "FREEZE movement",
    ],
    "LAW": [
        "Economic Bill of Rights",
        "Tax Reform Act",
        "Anti-ballistic Missile Treaty",
    ],
}


def normalize(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def extract_entities(df: pd.DataFrame, label_col: str = "llm_label") -> pd.DataFrame:
    labels = df[label_col].astype(str).tolist()
    words = df["word"].astype(str).tolist()
    rows = []
    i = 0
    while i < len(words):
        lab = labels[i]
        if lab == "none":
            i += 1
            continue
        j = i + 1
        while j < len(words) and labels[j] == lab:
            j += 1
        rows.append({"entity": normalize(" ".join(words[i:j])), "ner_type": lab})
        i = j
    if not rows:
        return pd.DataFrame(columns=["entity", "ner_type"])
    return pd.DataFrame(rows).drop_duplicates()


def eval_system(entity_df: pd.DataFrame, system: str) -> tuple[list[dict], dict]:
    by_type = []
    tp_total = fp_total = fn_total = 0
    for ner_type, gold_list in GOLD_DICT.items():
        gold = {normalize(x) for x in gold_list}
        pred = set(entity_df[entity_df["ner_type"] == ner_type]["entity"])
        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        by_type.append(
            {
                "system": system,
                "ner_type": ner_type,
                "precision": p,
                "recall": r,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "gold_support": len(gold),
                "pred_support": len(pred),
            }
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn

    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    overall = {
        "system": system,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": sum(x["f1"] for x in by_type) / len(by_type),
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
    }
    return by_type, overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Reagan entity extraction against dictionary-style gold.")
    parser.add_argument("--asr", required=True)
    parser.add_argument("--coref", required=True)
    parser.add_argument("--ner", required=True)
    parser.add_argument("--coref-ner", required=True)
    parser.add_argument("--output-dir", default="data/processed/reagan_dictionary_eval")
    args = parser.parse_args()

    systems = {
        "ASR": Path(args.asr),
        "Coref": Path(args.coref),
        "NER": Path(args.ner),
        "Coref_NER": Path(args.coref_ner),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_type_rows = []
    overall_rows = []
    for name, path in systems.items():
        df = pd.read_csv(path)
        ents = extract_entities(df)
        ents.to_csv(out_dir / f"{name.lower()}_entities.csv", index=False)
        bt, ov = eval_system(ents, name)
        by_type_rows.extend(bt)
        overall_rows.append(ov)

    pd.DataFrame(by_type_rows).to_csv(out_dir / "dictionary_eval_by_type.csv", index=False)
    pd.DataFrame(overall_rows).to_csv(out_dir / "dictionary_eval_overall.csv", index=False)
    print(f"Saved dictionary evaluation to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
