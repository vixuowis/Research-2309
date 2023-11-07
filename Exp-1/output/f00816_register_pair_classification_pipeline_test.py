from f00816_register_pair_classification_pipeline import *
def test_register_pair_classification_pipeline():
    # Test case 1
    register_pair_classification_pipeline()
    assert PIPELINE_REGISTRY.get_pipeline("pair-classification") == PairClassificationPipeline
    assert PIPELINE_REGISTRY.get_model("pair-classification", "pt") == AutoModelForSequenceClassification
    assert PIPELINE_REGISTRY.get_model("pair-classification", "tf") == TFAutoModelForSequenceClassification

    # Test case 2
    # ...
    # Test case 3
    # ...
    # Test case 4
    # ...
    # Test case 5
    # ...
