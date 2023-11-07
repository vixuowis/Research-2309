from f00415_load_bert_model import *
def test_load_bert_model():
    model = load_bert_model()
    assert isinstance(model, TFAutoModelForMultipleChoice)


test_load_bert_model()
