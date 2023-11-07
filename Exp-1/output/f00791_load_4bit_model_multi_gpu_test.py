from f00791_load_4bit_model_multi_gpu import *
def test_load_4bit_model_multi_gpu():
    model_name = "bigscience/bloom-2b5"
    model_4bit = load_4bit_model_multi_gpu(model_name)
    assert isinstance(model_4bit, AutoModelForCausalLM)

test_load_4bit_model_multi_gpu()
