from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

from jedm_pipeline.config import load_config
from jedm_pipeline.paths import resolve_path
from jedm_pipeline.topic_pyldavis import run_lda_pyldavis_for_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RQ3 LDA and export PyLDAvis HTML.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--num-topics", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    base_out = resolve_path(config, "topic_results_dir")
    out_dir = base_out / args.output_subdir
    num_topics = args.num_topics if args.num_topics is not None else int(config["rq3"]["num_topics"])

    run_lda_pyldavis_for_dir(
        input_dir=Path(args.input_dir),
        out_dir=out_dir,
        num_topics=num_topics,
        random_state=args.seed,
    )
    print(f"Saved PyLDAvis outputs to: {out_dir}")
    print(f"Open: {out_dir / 'lda_vis.html'}")


if __name__ == "__main__":
    main()
