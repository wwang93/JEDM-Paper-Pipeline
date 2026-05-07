from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABELS = ["PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "LAW"]


def normalize_label(value: str) -> str:
    s = str(value).strip()
    m = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "ORG": "ORG",
        "GPE": "GPE",
        "LOC": "LOC",
        "NORP": "NORP",
        "EVENT": "EVENT",
        "LAW": "LAW",
        "none": "none",
        "NONE": "none",
        "": "none",
    }
    return m.get(s, m.get(s.upper(), "none"))


def extract_entities(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    labels = [normalize_label(v) for v in df[label_col].tolist()]
    words = df["word"].astype(str).tolist()
    entities = []
    i = 0
    while i < len(words):
        lab = labels[i]
        if lab == "none" or lab not in LABELS:
            i += 1
            continue
        start = i
        j = i + 1
        while j < len(words) and labels[j] == lab:
            j += 1
        text = " ".join(words[start:j]).strip().lower()
        entities.append({"entity": text, "ner_type": lab})
        i = j
    if not entities:
        return pd.DataFrame(columns=["entity", "ner_type"])
    return pd.DataFrame(entities).drop_duplicates()


def score(pred: pd.DataFrame, gold: pd.DataFrame, system: str) -> tuple[dict, list[dict]]:
    rows = []
    tp_total = fp_total = fn_total = 0
    for t in LABELS:
        pset = set(pred[pred["ner_type"] == t]["entity"])
        gset = set(gold[gold["ner_type"] == t]["entity"])
        tp = len(pset & gset)
        fp = len(pset - gset)
        fn = len(gset - pset)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"system": system, "ner_type": t, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "gold_support": len(gset), "pred_support": len(pset)})
        tp_total += tp
        fp_total += fp
        fn_total += fn

    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    macro_f1 = sum(r["f1"] for r in rows) / len(rows)
    overall = {
        "system": system,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
    }
    return overall, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Entity-level PRF comparison against Reagan consensus gold.")
    parser.add_argument("--gold-csv", default="data/processed/reagan_consensus_prep/reagan_compare_cleaned.csv")
    parser.add_argument("--gold-label-col", default="consensus_label")
    parser.add_argument("--pred-asr", required=True)
    parser.add_argument("--pred-coref", required=True)
    parser.add_argument("--pred-ner", required=True)
    parser.add_argument("--pred-coref-ner", required=True)
    parser.add_argument("--output-dir", default="data/processed/reagan_entity_level_eval")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_df = pd.read_csv(args.gold_csv).head(600).copy()
    gold_entities = extract_entities(gold_df, args.gold_label_col)
    gold_entities.to_csv(out_dir / "gold_entities.csv", index=False)

    systems = {
        "ASR": args.pred_asr,
        "Coref": args.pred_coref,
        "NER": args.pred_ner,
        "Coref_NER": args.pred_coref_ner,
    }

    overall_rows = []
    type_rows = []
    for name, path in systems.items():
        pred_df = pd.read_csv(path)
        pred_entities = extract_entities(pred_df, "llm_label")
        pred_entities.to_csv(out_dir / f"{name.lower()}_entities.csv", index=False)
        overall, by_type = score(pred_entities, gold_entities, name)
        overall_rows.append(overall)
        type_rows.extend(by_type)

    pd.DataFrame(overall_rows).to_csv(out_dir / "entity_level_overall_metrics.csv", index=False)
    pd.DataFrame(type_rows).to_csv(out_dir / "entity_level_by_type_metrics.csv", index=False)
    print(f"Saved entity-level comparison outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
