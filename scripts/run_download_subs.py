from __future__ import annotations

import argparse

from _bootstrap import setup_src_path

setup_src_path()

from jedm_pipeline.pipeline import run_download_subtitles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_download_subtitles(args.config)
    print("Subtitles downloaded successfully.")


if __name__ == "__main__":
    main()
