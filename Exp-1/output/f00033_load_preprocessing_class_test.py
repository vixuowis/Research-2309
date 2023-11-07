from f00033_load_preprocessing_class import *
def test_load_preprocessing_class():
    model_name = "distilbert-base-uncased"
    preprocessing_class = load_preprocessing_class(model_name)

    assert isinstance(preprocessing_class, AutoTokenizer)


test_load_preprocessing_class()
