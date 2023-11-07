from f00888_initialize_model import *
def test_initialize_model():
    model = initialize_model()
    assert isinstance(model, TFTapasForQuestionAnswering)

    # Add more test cases here
    # ...


test_initialize_model()
