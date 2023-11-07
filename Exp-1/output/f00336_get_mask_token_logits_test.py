from f00336_get_mask_token_logits import *
def test_get_mask_token_logits():
	model = AutoModelForMaskedLM.from_pretrained("stevhliu/my_awesome_eli5_mlm_model")
	inputs = {}
	mask_token_index = 0
	mask_token_logits = get_mask_token_logits(model, inputs, mask_token_index)
	assert isinstance(mask_token_logits, torch.Tensor), "mask_token_logits should be a torch.Tensor"

	# Add more test cases here

if __name__ == '__main__':
	test_get_mask_token_logits()

