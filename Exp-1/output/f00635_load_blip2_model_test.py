from f00635_load_blip2_model import *
def test_load_blip2_model():
    model = load_blip2_model()
    assert isinstance(model, Blip2ForConditionalGeneration)
    print("Model loaded successfully")

test_load_blip2_model()
