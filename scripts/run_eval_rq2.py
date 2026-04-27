from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

from jedm_pipeline.config import load_config
from jedm_pipeline.metrics_ner import compare_ner_systems, save_ner_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--gold-csv", required=True)
    parser.add_argument("--asr-csv", required=True)
    parser.add_argument("--coref-csv", required=True)
    parser.add_argument("--ner-csv", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(config["_project_root"]) / "data" / "results" / "rq2_ner_metrics"
    system_csvs = {
        "ASR": Path(args.asr_csv),
        "LLMs_Coref": Path(args.coref_csv),
        "LLMs_NER": Path(args.ner_csv),
    }
    summary_df, by_type_df = compare_ner_systems(system_csvs, Path(args.gold_csv))
    save_ner_results(summary_df, by_type_df, out_dir)
    print(summary_df.to_string(index=False))
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
