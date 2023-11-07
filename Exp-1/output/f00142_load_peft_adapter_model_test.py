from f00142_load_peft_adapter_model import *
def test_load_peft_adapter_model():
	peft_model_id = "ybelkada/opt-350m-lora"
	model = load_peft_adapter_model(peft_model_id)
	# Add test cases here

