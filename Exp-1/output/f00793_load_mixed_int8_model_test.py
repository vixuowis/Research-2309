from f00793_load_mixed_int8_model import *
def test_load_mixed_int8_model():
    model_name = "bigscience/bloom-2b5"
    model = load_mixed_int8_model(model_name)
    assert isinstance(model, AutoModelForCausalLM)


test_load_mixed_int8_model()
