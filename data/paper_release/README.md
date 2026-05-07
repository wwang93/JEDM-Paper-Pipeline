# Paper Release Data (Current Revision)

This folder contains paper-facing CSV outputs used in the current manuscript revision.

## Files

- `reagan_llm_eval_model_comparison_full_metrics.csv`  
  Model comparison on the 600-token human-consensus benchmark (`gpt-4o` vs `gpt-5.5`).

- `reagan_600_per_type_comparison_gpt4o_vs_gpt55.csv`  
  Token-level per-type precision/recall/F1 side-by-side comparison.

- `rq2_entity_profile_mention_summary.csv`  
  Main RQ2 corpus profile table (mention-level) across all 48 ASR transcripts.

- `rq2_entity_profile_token_summary.csv`  
  Token-level variant of the corpus profile table.

- `sample4_run_summary.csv`  
  Four-episode API trial token-volume summary for cost estimation.

- `sample4_vs_asr_llm_gold_overall.csv`  
  Four-episode overall entity-level comparison against the LLM-ASR operational benchmark.

- `sample4_vs_asr_llm_gold_by_type.csv`  
  Four-episode by-type entity-level comparison against the LLM-ASR operational benchmark.

## Notes

- These files are manuscript-support artifacts for reproducibility and reporting.
- Large local/intermediate files remain excluded by `.gitignore`.
