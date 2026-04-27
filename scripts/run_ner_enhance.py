from __future__ import annotations

import argparse

from _bootstrap import setup_src_path

setup_src_path()

from jedm_pipeline.pipeline import run_ner_enhance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    processed, skipped = run_ner_enhance(args.config)
    print(f"NER enhancement done. processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
