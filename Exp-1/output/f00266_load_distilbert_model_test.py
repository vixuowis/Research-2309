from f00266_load_distilbert_model import *
def test_load_distilbert_model():
    model_name = 'distilbert-base-uncased'
    model = load_distilbert_model(model_name)
    assert isinstance(model, TFAutoModelForQuestionAnswering)

    model_name = 'distilbert-base-cased'
    model = load_distilbert_model(model_name)
    assert isinstance(model, TFAutoModelForQuestionAnswering)

    model_name = 'distilbert-base-multilingual-cased'
    model = load_distilbert_model(model_name)
    assert isinstance(model, TFAutoModelForQuestionAnswering)

    model_name = 'distilbert-base-uncased-distilled-squad'
    model = load_distilbert_model(model_name)
    assert isinstance(model, TFAutoModelForQuestionAnswering)

    model_name = 'distilbert-base-cased-distilled-squad'
    model = load_distilbert_model(model_name)
    assert isinstance(model, TFAutoModelForQuestionAnswering)
