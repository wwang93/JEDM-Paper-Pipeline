from __future__ import annotations

import argparse

from _bootstrap import setup_src_path

setup_src_path()

from jedm_pipeline.pipeline import run_srt_to_txt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    count = run_srt_to_txt(args.config)
    print(f"Converted {count} .srt files to .txt.")


if __name__ == "__main__":
    main()
