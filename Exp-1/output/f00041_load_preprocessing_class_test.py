from f00041_load_preprocessing_class import *
def test_load_preprocessing_class():
    tokenizer = load_preprocessing_class(AutoTokenizer, "distilbert-base-uncased")
    assert isinstance(tokenizer, AutoTokenizer)

# Test the function

test_load_preprocessing_class()
