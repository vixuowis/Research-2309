from f00298_load_distil_gpt2 import *
def test_load_distil_gpt2():
    model = load_distil_gpt2()

    assert isinstance(model, TFAutoModelForCausalLM)
    assert model.config.architectures[0] == "GPT2LMHeadModel"
    assert model.config.model_type == "gpt2"

    print("All tests pass.")
