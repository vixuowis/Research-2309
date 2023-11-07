from f00009_load_model_and_tokenizer import *
def test_load_model_and_tokenizer():
    model_name = "bert-base-uncased"
    model, tokenizer = load_model_and_tokenizer(model_name)

    assert isinstance(model, AutoModelForSequenceClassification)
    assert isinstance(tokenizer, AutoTokenizer)

    # Additional test cases
    # ...


test_load_model_and_tokenizer()
