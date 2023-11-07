from f00850_load_model import *
def test_load_model():
    model_name = 'bigscience/bloom'
    model = load_model(model_name)
    assert isinstance(model, AutoModelForCausalLM)
    print('Model loaded successfully!')

test_load_model()
