from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

from jedm_pipeline.config import load_config
from jedm_pipeline.io_utils import read_text
from jedm_pipeline.llm_client import OpenAITextEnhancer, batch_enhance_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NER enhancement on a specified input directory.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true", help="Reprocess files even if outputs already exist")
    args = parser.parse_args()

    config = load_config(args.config)
    enhancer = OpenAITextEnhancer(
        api_key=config["env"]["OPENAI_API_KEY"],
        model=config["llm"]["model"],
        max_chars=config["llm"]["max_chars"],
        sleep_seconds=config["llm"]["sleep_seconds"],
        max_retries=config["llm"]["max_retries"],
    )

    prompt_path = Path(config["_project_root"]) / "prompts" / "ner_prompt.txt"
    prompt_template = read_text(prompt_path)

    processed, skipped = batch_enhance_dir(
        enhancer=enhancer,
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        prompt_template=prompt_template,
        system_prompt="You are an expert in American history and advanced NER.",
        skip_existing=not args.force,
    )

    print(f"NER-on-dir done. processed={processed}, skipped={skipped}")
    print(f"Saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
