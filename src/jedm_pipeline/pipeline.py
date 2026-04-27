from __future__ import annotations

import subprocess
from pathlib import Path

from .config import load_config
from .io_utils import read_text
from .llm_client import OpenAITextEnhancer, batch_enhance_dir
from .paths import resolve_path
from .text_cleaning import batch_convert_srt_dir


def run_download_subtitles(config_path: str = "configs/default.yaml") -> None:
    config = load_config(config_path)
    out_dir = resolve_path(config, "raw_srt_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--ignore-errors",
        "--yes-playlist",
        "--write-subs",
        "--sub-lang",
        config["download"]["subtitle_lang"],
        "--sub-format",
        config["download"]["subtitle_format"],
        "--skip-download",
        "-o",
        str(out_dir / "%(title)s.%(ext)s"),
        config["download"]["playlist_url"],
    ]
    subprocess.run(cmd, check=True)


def run_srt_to_txt(config_path: str = "configs/default.yaml") -> int:
    config = load_config(config_path)
    srt_dir = resolve_path(config, "raw_srt_dir")
    txt_dir = resolve_path(config, "raw_txt_dir")
    return batch_convert_srt_dir(srt_dir, txt_dir)


def _run_llm_step(config: dict, in_key: str, out_key: str, prompt_file: Path, system_prompt: str) -> tuple[int, int]:
    enhancer = OpenAITextEnhancer(
        api_key=config["env"]["OPENAI_API_KEY"],
        model=config["llm"]["model"],
        max_chars=config["llm"]["max_chars"],
        sleep_seconds=config["llm"]["sleep_seconds"],
        max_retries=config["llm"]["max_retries"],
    )
    prompt_template = read_text(prompt_file)
    in_dir = resolve_path(config, in_key)
    out_dir = resolve_path(config, out_key)
    return batch_enhance_dir(
        enhancer=enhancer,
        input_dir=in_dir,
        output_dir=out_dir,
        prompt_template=prompt_template,
        system_prompt=system_prompt,
        skip_existing=config["llm"]["skip_existing"],
    )


def run_coref(config_path: str = "configs/default.yaml") -> tuple[int, int]:
    config = load_config(config_path)
    prompt_path = Path(config["_project_root"]) / "prompts" / "coref_prompt.txt"
    return _run_llm_step(
        config,
        in_key="raw_txt_dir",
        out_key="coref_txt_dir",
        prompt_file=prompt_path,
        system_prompt="You are an assistant skilled in coreference resolution.",
    )


def run_ner_enhance(config_path: str = "configs/default.yaml") -> tuple[int, int]:
    config = load_config(config_path)
    prompt_path = Path(config["_project_root"]) / "prompts" / "ner_prompt.txt"
    return _run_llm_step(
        config,
        in_key="raw_txt_dir",
        out_key="ner_txt_dir",
        prompt_file=prompt_path,
        system_prompt="You are an expert in American history and advanced NER.",
    )
