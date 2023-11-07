from f00335_tokenize_text import *
def test_tokenize_text():
	text = "This is a sample text"
	inputs, mask_token_index = tokenize_text(text)
	assert isinstance(inputs, dict)
	assert isinstance(mask_token_index, torch.Tensor)
	assert len(mask_token_index) > 0

	# Additional test cases
	text = "Another sample text"
	inputs, mask_token_index = tokenize_text(text)
	assert isinstance(inputs, dict)
	assert isinstance(mask_token_index, torch.Tensor)
	assert len(mask_token_index) > 0

	text = "Yet another sample text"
	inputs, mask_token_index = tokenize_text(text)
	assert isinstance(inputs, dict)
	assert isinstance(mask_token_index, torch.Tensor)
	assert len(mask_token_index) > 0

	print("All test cases pass")
