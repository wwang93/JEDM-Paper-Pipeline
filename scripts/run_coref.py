from __future__ import annotations

import argparse

from _bootstrap import setup_src_path

setup_src_path()

from jedm_pipeline.pipeline import run_coref


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    processed, skipped = run_coref(args.config)
    print(f"Coref done. processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
