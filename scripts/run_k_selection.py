from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

from jedm_pipeline.topic_selection import evaluate_k_range, save_k_selection_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible K selection for LDA.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (0..seeds-1)")
    parser.add_argument("--top-n-terms", type=int, default=10)
    parser.add_argument("--coherence-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    k_values = list(range(args.k_min, args.k_max + 1))
    seed_values = list(range(args.seeds))

    result = evaluate_k_range(
        input_dir=Path(args.input_dir),
        k_values=k_values,
        seeds=seed_values,
        top_n_terms=args.top_n_terms,
        coherence_tolerance=args.coherence_tolerance,
    )
    save_k_selection_outputs(result, Path(args.output_dir))

    print(result.summary_df.to_string(index=False))
    print(f"Selected K: {result.selected_k}")
    print(f"Saved outputs: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
