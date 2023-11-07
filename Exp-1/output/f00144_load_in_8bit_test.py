from f00144_load_in_8bit import *
def test_load_in_8bit():
    peft_model_id = "ybelkada/opt-350m-lora"
    model = load_in_8bit(peft_model_id)

    assert isinstance(model, AutoModelForCausalLM)

    # Add more test cases here


test_load_in_8bit()
