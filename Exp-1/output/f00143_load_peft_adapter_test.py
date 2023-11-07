from f00143_load_peft_adapter import *
def test_load_peft_adapter():
    model_id = "facebook/opt-350m"
    peft_model_id = "ybelkada/opt-350m-lora"
    model = load_peft_adapter(model_id, peft_model_id)

    assert isinstance(model, AutoModelForCausalLM)
    assert model.config.id2label[0] == 'LABEL_0'
    assert model.config.label2id['LABEL_0'] == 0
    assert model.config.id2label[1] == 'LABEL_1'
    assert model.config.label2id['LABEL_1'] == 1
    assert model.config.id2label[2] == 'LABEL_2'
    assert model.config.label2id['LABEL_2'] == 2

    print("All test cases pass.")
