from f00215_tokenize_text import *
def test_tokenize_text():
    assert tokenize_text("This is a test text.") == {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}
