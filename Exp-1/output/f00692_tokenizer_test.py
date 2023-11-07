from f00692_tokenizer import *
def test_tokenizer():
    assert tokenizer(chinese_text, return_tensors) == encoded_zh


test_tokenizer()
