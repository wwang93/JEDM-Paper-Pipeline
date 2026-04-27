from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .io_utils import ensure_dir, iter_txt_files, read_text


def _basic_tokenizer(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{3,}", text.lower())


def _load_documents(input_dir: Path) -> tuple[list[str], list[str]]:
    files = iter_txt_files(input_dir)
    docs = [read_text(path) for path in files]
    names = [path.name for path in files]
    if not docs:
        raise ValueError(f"No txt files found in {input_dir}")
    return docs, names


def _topic_terms(model: LatentDirichletAllocation, vectorizer: CountVectorizer, topn: int = 12) -> pd.DataFrame:
    vocab = vectorizer.get_feature_names_out()
    rows: list[dict[str, str | int]] = []
    for idx, topic_weights in enumerate(model.components_):
        top_idx = topic_weights.argsort()[-topn:][::-1]
        words = [vocab[i] for i in top_idx]
        rows.append({"topic": idx, "top_terms": ", ".join(words)})
    return pd.DataFrame(rows)


def _plot_topic_proportions(doc_topic_df: pd.DataFrame, out_path: Path) -> None:
    topic_cols = [c for c in doc_topic_df.columns if c.startswith("topic_")]
    mean_props = doc_topic_df[topic_cols].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    mean_props.plot(kind="bar")
    plt.title("Average Topic Proportions")
    plt.ylabel("Mean proportion")
    plt.xlabel("Topic")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_lda_for_dir(input_dir: Path, out_dir: Path, num_topics: int = 5) -> None:
    ensure_dir(out_dir)
    docs, names = _load_documents(input_dir)

    vectorizer = CountVectorizer(
        tokenizer=_basic_tokenizer,
        stop_words="english",
        lowercase=True,
        min_df=2,
        max_df=0.95,
    )
    dtm = vectorizer.fit_transform(docs)

    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        learning_method="batch",
        max_iter=30,
    )
    doc_topic = lda.fit_transform(dtm)

    topics_df = _topic_terms(lda, vectorizer, topn=12)
    topics_df.to_csv(out_dir / "topic_top_terms.csv", index=False)

    dominant = pd.DataFrame(doc_topic)
    dominant.columns = [f"topic_{i}" for i in range(num_topics)]
    dominant.insert(0, "file", names)
    dominant["dominant_topic"] = dominant[[f"topic_{i}" for i in range(num_topics)]].idxmax(axis=1)
    dominant.to_csv(out_dir / "doc_topic_distribution.csv", index=False)

    _plot_topic_proportions(dominant, out_dir / "topic_proportions.png")
