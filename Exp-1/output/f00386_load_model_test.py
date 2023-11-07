from f00386_load_model import *
def test_load_model():
    assert isinstance(load_model('checkpoint'), TFAutoModelForSeq2SeqLM)
    assert isinstance(load_model('/path/to/checkpoint'), TFAutoModelForSeq2SeqLM)
    assert isinstance(load_model('model_name'), TFAutoModelForSeq2SeqLM)
    assert isinstance(load_model('model_name.pt'), TFAutoModelForSeq2SeqLM)
    assert isinstance(load_model('model_name.bin'), TFAutoModelForSeq2SeqLM)
