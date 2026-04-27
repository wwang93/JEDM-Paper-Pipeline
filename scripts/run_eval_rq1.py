from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

from jedm_pipeline.config import load_config
from jedm_pipeline.metrics_asr import evaluate_asr_files, save_asr_results
from jedm_pipeline.paths import resolve_path


def _build_file_map(raw_dir: Path, human_file: str) -> dict[str, Path]:
    files = {p.stem: p for p in raw_dir.glob("*.txt")}
    if human_file not in files:
        raise FileNotFoundError(f"Human file key '{human_file}' not found in {raw_dir}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = resolve_path(config, "raw_txt_dir")
    out_dir = Path(config["_project_root"]) / "data" / "results" / "rq1_asr_metrics"

    human_key = config["rq1"]["human_key"]
    compare_keys = config["rq1"]["compare_keys"]
    file_map = _build_file_map(raw_dir, human_key)

    df = evaluate_asr_files(file_map, human_key, compare_keys)
    save_asr_results(df, out_dir)
    print(df.to_string(index=False))
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
