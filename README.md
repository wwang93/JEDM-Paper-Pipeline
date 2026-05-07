# JEDM-Paper-Pipeline
JEDM Speical Issue for LLMs in Qual Research: LLM-Assisted Transcript Preprocessing for Improved Entity Recognition and Qualitative Analysis: A Case Study with Crash Course US History Videos

## Local Reproducible Pipeline (No Notebook)

This repository now includes a local Python pipeline under `src/jedm_pipeline` and runnable scripts under `scripts`.

### Repository Scope

This GitHub repository keeps code, prompts, configs, and reproducible scripts that are directly tied to the paper workflow.

- Included: pipeline code in `src/`, execution scripts in `scripts/`, and reproducibility docs/configs.
- Kept local only: manuscript drafts, exploratory artifacts, and local data outputs (see `.gitignore` and `_local_extensions/`).

### Paper-Facing Data Files

Current manuscript-facing CSV outputs are versioned in `data/paper_release/`.
See `data/paper_release/README.md` for file-level descriptions.

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2) Configure

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

Edit `configs/default.yaml` for your local input/output paths.

### 3) Run

```bash
python scripts/run_srt_to_txt.py --config configs/default.yaml
python scripts/run_coref.py --config configs/default.yaml
python scripts/run_ner_enhance.py --config configs/default.yaml
```

Or run all preprocessing steps:

```bash
python scripts/run_all.py --config configs/default.yaml
```

### 4) Evaluate

RQ1 (ASR metrics):

```bash
python scripts/run_eval_rq1.py --config configs/default.yaml
```

RQ2 (entity-level NER metrics):

```bash
python scripts/run_eval_rq2.py --config configs/default.yaml --gold-csv path/to/GoldNER.csv --asr-csv path/to/ASRNER.csv --coref-csv path/to/LLMscore.csv --ner-csv path/to/LLMsNER.csv
```

RQ2 (revised two-step workflow used in manuscript):

1) Validate model coding quality on the 600-token human-consensus benchmark (compare models):

```bash
python scripts/run_reagan_token_eval.py --config configs/default.yaml --model gpt-4o --n 600 --output-dir data/processed/reagan_llm_eval
python scripts/run_reagan_token_eval.py --config configs/default.yaml --model gpt-5.5 --n 600 --output-dir data/processed/reagan_llm_eval_gpt55
```

2) Apply the selected model to all 48 ASR transcripts and produce corpus-level entity profile summaries:

```bash
python scripts/run_full_corpus_token_annotation.py --config configs/default.yaml --model gpt-5.5 --input-dir "data/external/paper_meta_data/Paper Meta Data/USHistory_raw_txts" --codebook-xlsx "data/Human_The_Reagan_Revolution_Crash_Course_US_History_43_en_one_column_words.xlsx" --output-dir data/processed/full_corpus_llm55_alltokens_v1
python scripts/run_entity_profile_summary.py --labels-csv data/processed/full_corpus_llm55_alltokens_v1/all_docs_token_labels.csv --output-dir data/processed/full_corpus_llm55_alltokens_v1
```

This produces:

- `rq2_entity_profile_mention_summary.csv` (used for Table 1 style corpus entity profile)
- `rq2_entity_profile_token_summary.csv` (token-count variant)

RQ3 topic modeling:

```bash
python scripts/run_eval_rq3_topic.py --config configs/default.yaml --input-dir data/interim/ner_txt --output-subdir ner_version
```

Corpus descriptives table (word/token/vocabulary across stages):

```bash
python scripts/run_corpus_descriptives.py --asr-dir "data/external/paper_meta_data/Paper Meta Data/USHistory_raw_txts" --coref-dir "data/external/paper_meta_data/Paper Meta Data/USHistory_LLMs_coreferenced_txts" --ner-dir "data/external/paper_meta_data/Paper Meta Data/USHistory_LLMsNER_txts" --output-dir "data/results/rq3_topic_modeling/corpus_descriptives"
```

Outputs: `episode_level_descriptives.csv` and `stage_summary_descriptives.csv`.

RQ3 with PyLDAvis (separate Python 3.11/3.12 env recommended):

```bash
python -m venv .venv-pyldavis
source .venv-pyldavis/bin/activate  # Windows: .venv-pyldavis\Scripts\activate
pip install -r requirements-pyldavis.txt
python scripts/run_eval_rq3_pyldavis.py --config configs/default.yaml --input-dir "data/external/paper_meta_data/Paper Meta Data/USHistory_raw_txts" --output-subdir raw_pyldavis
```

This writes `lda_vis.html` plus topic CSV files under `data/results/rq3_topic_modeling/<output-subdir>/`.

### Reproducible K Selection for LDA (Reviewer Response)

Use this script to replace subjective "iterative inspection" with a reproducible protocol.

```bash
python scripts/run_k_selection.py --input-dir "data/external/paper_meta_data/Paper Meta Data/USHistory_raw_txts" --output-dir "data/results/rq3_topic_modeling/raw_k_selection" --k-min 3 --k-max 10 --seeds 5 --top-n-terms 10
```

Outputs:

- `k_selection_runs.csv`: one row per `(K, seed)` with NPMI coherence, diversity, perplexity
- `k_selection_summary.csv`: aggregated means/std and stability per K
- `k_selection_diagnostics.png`: coherence/diversity/stability curves

Protocol used:

1. Scan `K=3..10`
2. Repeat each K with 5 random seeds
3. Compute NPMI coherence, topic diversity, and cross-seed topic stability
4. Select K from the high-coherence region while favoring stability and smaller K
