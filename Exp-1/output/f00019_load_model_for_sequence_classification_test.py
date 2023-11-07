from f00019_load_model_for_sequence_classification import *
def test_load_model_for_sequence_classification():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    model = load_model_for_sequence_classification(model_name)
    assert isinstance(model, TFAutoModelForSequenceClassification)


test_load_model_for_sequence_classification()
