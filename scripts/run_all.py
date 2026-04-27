from __future__ import annotations

import argparse

from _bootstrap import setup_src_path

setup_src_path()

from jedm_pipeline.pipeline import run_coref, run_ner_enhance, run_srt_to_txt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local JEDM pipeline (without notebook).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-coref", action="store_true")
    parser.add_argument("--skip-ner", action="store_true")
    args = parser.parse_args()

    converted = run_srt_to_txt(args.config)
    print(f"SRT->TXT converted: {converted}")

    if not args.skip_coref:
        processed, skipped = run_coref(args.config)
        print(f"Coref: processed={processed}, skipped={skipped}")

    if not args.skip_ner:
        processed, skipped = run_ner_enhance(args.config)
        print(f"NER: processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
