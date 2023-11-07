from f00733_load_custom_model import *
def test_load_custom_model():
    model = load_custom_model()
    assert isinstance(model, AutoModelForImageClassification)
    print("Custom model loaded successfully.")
