from f00869_instantiate_model import *
def test_instantiate_model():
    model = instantiate_model()
    assert isinstance(model, UperNetForSemanticSegmentation)
    print("Test Passed")
