from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

from jedm_pipeline.config import load_config
from jedm_pipeline.paths import resolve_path
from jedm_pipeline.topic_modeling import run_lda_for_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-subdir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    base_out = resolve_path(config, "topic_results_dir")
    out_dir = base_out / args.output_subdir
    run_lda_for_dir(Path(args.input_dir), out_dir, num_topics=config["rq3"]["num_topics"])
    print(f"Saved topic outputs to: {out_dir}")


if __name__ == "__main__":
    main()
