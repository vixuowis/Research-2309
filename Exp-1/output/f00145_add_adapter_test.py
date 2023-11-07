from f00145_add_adapter import *
def test_add_adapter():
    model_id = "facebook/opt-350m"
    model = AutoModelForCausalLM.from_pretrained(model_id)

    lora_config = LoraConfig(
        target_modules=["q_proj", "k_proj"],
        init_lora_weights=False
    )

    adapter_name = model.add_adapter(lora_config, adapter_name="adapter_1")
    assert adapter_name == "adapter_1"

    # Add additional test cases here


test_add_adapter()
