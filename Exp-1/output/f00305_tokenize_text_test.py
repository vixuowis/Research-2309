from f00305_tokenize_text import *
def test_tokenize_text():
	text = 'This is a sample text'
	model_name = 'my_awesome_eli5_clm-model'
	expected_output = torch.tensor([[101, 2023, 2003, 1037, 7099, 3793, 102]])
	output = tokenize_text(text, model_name)
	assert torch.equal(output, expected_output), f'Expected {expected_output}, but got {output}'

	text = 'Another example'
	expected_output = torch.tensor([[101, 2178, 2742, 102]])
	output = tokenize_text(text, model_name)
	assert torch.equal(output, expected_output), f'Expected {expected_output}, but got {output}'

	text = 'Yet another example'
	expected_output = torch.tensor([[101, 2664, 2178, 2742, 102]])
	output = tokenize_text(text, model_name)
	assert torch.equal(output, expected_output), f'Expected {expected_output}, but got {output}'

	text = 'One more example'
	expected_output = torch.tensor([[101, 2028, 2062, 2742, 102]])
	output = tokenize_text(text, model_name)
	assert torch.equal(output, expected_output), f'Expected {expected_output}, but got {output}'

	text = 'Final example'
	expected_output = torch.tensor([[101, 2345, 2742, 102]])
	output = tokenize_text(text, model_name)
	assert torch.equal(output, expected_output), f'Expected {expected_output}, but got {output}'

print('All test cases pass')

test_tokenize_text()
