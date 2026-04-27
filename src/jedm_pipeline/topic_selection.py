from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .io_utils import ensure_dir, iter_txt_files, read_text


def _basic_tokenizer(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{3,}", text.lower())


def _load_documents(input_dir: Path) -> list[str]:
    files = iter_txt_files(input_dir)
    docs = [read_text(path) for path in files]
    if not docs:
        raise ValueError(f"No txt files found in {input_dir}")
    return docs


def _build_vectorizer() -> CountVectorizer:
    return CountVectorizer(
        tokenizer=_basic_tokenizer,
        stop_words="english",
        lowercase=True,
        min_df=2,
        max_df=0.95,
        max_features=5000,
    )


def _topic_word_indices(lda: LatentDirichletAllocation, top_n: int) -> list[np.ndarray]:
    topics: list[np.ndarray] = []
    for topic_weights in lda.components_:
        top_idx = np.argsort(topic_weights)[-top_n:][::-1]
        topics.append(top_idx)
    return topics


def _topic_diversity(topic_indices: list[np.ndarray], top_n: int) -> float:
    all_terms = np.concatenate(topic_indices)
    return float(len(np.unique(all_terms)) / (len(topic_indices) * top_n))


def _npmi_topic_coherence(topic_terms: np.ndarray, x_binary, doc_count: int) -> float:
    if len(topic_terms) < 2:
        return 0.0
    x_sub = x_binary[:, topic_terms]
    df = np.asarray(x_sub.sum(axis=0)).ravel()
    cooc = (x_sub.T @ x_sub).toarray()

    scores: list[float] = []
    for i in range(len(topic_terms)):
        for j in range(i + 1, len(topic_terms)):
            cij = cooc[i, j]
            if cij <= 0:
                continue
            pi = df[i] / doc_count
            pj = df[j] / doc_count
            pij = cij / doc_count
            pmi = math.log(pij / (pi * pj + 1e-12) + 1e-12)
            npmi = pmi / (-math.log(pij + 1e-12))
            scores.append(npmi)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def _model_npmi(topic_indices: list[np.ndarray], x_binary, doc_count: int) -> float:
    topic_scores = [_npmi_topic_coherence(terms, x_binary, doc_count) for terms in topic_indices]
    return float(np.mean(topic_scores))


def _topic_stability(seed_topics: list[list[np.ndarray]]) -> float:
    if len(seed_topics) < 2:
        return 0.0

    pair_scores: list[float] = []
    for a, b in combinations(seed_topics, 2):
        k = len(a)
        sim = np.zeros((k, k), dtype=float)
        for i in range(k):
            set_i = set(a[i].tolist())
            for j in range(k):
                set_j = set(b[j].tolist())
                inter = len(set_i & set_j)
                union = len(set_i | set_j)
                sim[i, j] = inter / union if union else 0.0
        row, col = linear_sum_assignment(1 - sim)
        pair_scores.append(float(sim[row, col].mean()))

    return float(np.mean(pair_scores))


@dataclass
class KSelectionResult:
    run_df: pd.DataFrame
    summary_df: pd.DataFrame
    selected_k: int


def evaluate_k_range(
    input_dir: Path,
    k_values: list[int],
    seeds: list[int],
    top_n_terms: int = 10,
    coherence_tolerance: float = 0.01,
) -> KSelectionResult:
    docs = _load_documents(input_dir)
    vectorizer = _build_vectorizer()
    x_counts = vectorizer.fit_transform(docs)
    x_binary = (x_counts > 0).astype(int)

    run_rows: list[dict[str, float | int]] = []
    topics_by_k: dict[int, list[list[np.ndarray]]] = {k: [] for k in k_values}

    for k in k_values:
        for seed in seeds:
            lda = LatentDirichletAllocation(
                n_components=k,
                random_state=seed,
                learning_method="batch",
                max_iter=30,
            )
            doc_topic = lda.fit_transform(x_counts)
            topic_indices = _topic_word_indices(lda, top_n=top_n_terms)
            topics_by_k[k].append(topic_indices)

            coherence = _model_npmi(topic_indices, x_binary=x_binary, doc_count=x_counts.shape[0])
            diversity = _topic_diversity(topic_indices, top_n=top_n_terms)
            perplexity = float(lda.perplexity(x_counts))
            max_topic_mass = float(np.max(doc_topic, axis=1).mean())

            run_rows.append(
                {
                    "k": k,
                    "seed": seed,
                    "npmi_coherence": coherence,
                    "topic_diversity": diversity,
                    "perplexity": perplexity,
                    "mean_max_topic_mass": max_topic_mass,
                }
            )

    run_df = pd.DataFrame(run_rows)

    summary_rows: list[dict[str, float | int]] = []
    for k in k_values:
        sub = run_df[run_df["k"] == k]
        stability = _topic_stability(topics_by_k[k])
        summary_rows.append(
            {
                "k": k,
                "npmi_coherence_mean": float(sub["npmi_coherence"].mean()),
                "npmi_coherence_std": float(sub["npmi_coherence"].std(ddof=0)),
                "topic_diversity_mean": float(sub["topic_diversity"].mean()),
                "topic_stability": stability,
                "perplexity_mean": float(sub["perplexity"].mean()),
                "mean_max_topic_mass": float(sub["mean_max_topic_mass"].mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("k").reset_index(drop=True)

    max_coh = float(summary_df["npmi_coherence_mean"].max())
    candidates = summary_df[summary_df["npmi_coherence_mean"] >= (max_coh - coherence_tolerance)].copy()
    candidates = candidates.sort_values(["topic_stability", "topic_diversity_mean", "k"], ascending=[False, False, True])
    selected_k = int(candidates.iloc[0]["k"])

    return KSelectionResult(run_df=run_df, summary_df=summary_df, selected_k=selected_k)


def save_k_selection_outputs(result: KSelectionResult, out_dir: Path) -> None:
    ensure_dir(out_dir)
    result.run_df.to_csv(out_dir / "k_selection_runs.csv", index=False)
    result.summary_df.to_csv(out_dir / "k_selection_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sdf = result.summary_df

    axes[0].plot(sdf["k"], sdf["npmi_coherence_mean"], marker="o")
    axes[0].fill_between(
        sdf["k"],
        sdf["npmi_coherence_mean"] - sdf["npmi_coherence_std"],
        sdf["npmi_coherence_mean"] + sdf["npmi_coherence_std"],
        alpha=0.2,
    )
    axes[0].set_title("Coherence (NPMI)")
    axes[0].set_xlabel("K")

    axes[1].plot(sdf["k"], sdf["topic_diversity_mean"], marker="o", color="tab:green")
    axes[1].set_title("Topic Diversity")
    axes[1].set_xlabel("K")

    axes[2].plot(sdf["k"], sdf["topic_stability"], marker="o", color="tab:orange")
    axes[2].set_title("Topic Stability")
    axes[2].set_xlabel("K")

    for ax in axes:
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle(f"K Selection Diagnostics (selected K={result.selected_k})")
    plt.tight_layout()
    plt.savefig(out_dir / "k_selection_diagnostics.png")
    plt.close()
