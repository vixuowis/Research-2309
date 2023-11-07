from f00847_load_model import *
def test_load_model():
    model = load_model()
    assert isinstance(model, AutoModelForSequenceClassification)
    print("Model loaded successfully!")


test_load_model()
