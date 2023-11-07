from typing import *
from pair_classification import PairClassificationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

def register_pair_classification_pipeline():
    """Register the PairClassificationPipeline in the pipeline registry

    This allows the pipeline to be accessed using `pipeline('pair-classification')`

    Args:
        None

    Returns:
        None
    """
    PIPELINE_REGISTRY.register_pipeline(
        "pair-classification",
        pipeline_class=PairClassificationPipeline,
        pt_model=AutoModelForSequenceClassification,
        tf_model=TFAutoModelForSequenceClassification,
    )
