from f00675_load_model import *
def test_load_model():
    checkpoint = "HuggingFaceM4/idefics-9b"
    model, tokenizer = load_model(checkpoint)
    assert isinstance(model, T5ForConditionalGeneration)
    assert isinstance(tokenizer, T5Tokenizer)


test_load_model()
