import pandas as pd

from jedm_pipeline.metrics_ner import entity_level_metrics


def test_entity_level_metrics_simple_case() -> None:
    pred_df = pd.DataFrame(
        {
            "entity": ["Ronald Reagan", "Congress"],
            "ner_type": ["PERSON", "ORG"],
        }
    )
    gold_df = pd.DataFrame(
        {
            "entity": ["Ronald Reagan", "NATO"],
            "ner_type": ["PERSON", "ORG"],
        }
    )
    result = entity_level_metrics(pred_df, gold_df, "ASR")
    assert result["TP"] == 1
    assert result["FP"] == 1
    assert result["FN"] == 1
