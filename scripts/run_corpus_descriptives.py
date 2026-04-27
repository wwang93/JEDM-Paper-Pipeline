from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

from jedm_pipeline.corpus_descriptives import build_episode_descriptives, build_stage_summary, save_descriptives


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corpus descriptives across transcript stages.")
    parser.add_argument("--asr-dir", required=True)
    parser.add_argument("--coref-dir", required=True)
    parser.add_argument("--ner-dir", required=True)
    parser.add_argument("--coref-ner-dir", default=None, help="Optional ASR->Coref->NER directory")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    stage_dirs = {
        "ASR": Path(args.asr_dir),
        "LLM_Coref": Path(args.coref_dir),
        "LLM_NER": Path(args.ner_dir),
    }
    if args.coref_ner_dir:
        stage_dirs["LLM_Coref_NER"] = Path(args.coref_ner_dir)

    episode_df = build_episode_descriptives(stage_dirs)
    summary_df = build_stage_summary(episode_df)
    save_descriptives(episode_df, summary_df, Path(args.output_dir))

    print(summary_df.to_string(index=False))
    print(f"Saved: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
