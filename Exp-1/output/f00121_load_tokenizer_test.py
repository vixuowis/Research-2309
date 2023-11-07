from f00121_load_tokenizer import *
def test_load_tokenizer():
    model_name = "bert-base-cased"
    tokenizer = load_tokenizer(model_name)
    assert isinstance(tokenizer, AutoTokenizer)

test_load_tokenizer()
