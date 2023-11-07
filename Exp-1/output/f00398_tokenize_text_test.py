from f00398_tokenize_text import *
def test_tokenize_text():
	text = "This is a sample text."
	inputs = tokenize_text(text)
	assert inputs.shape == (1, 9)
	assert inputs[0, 0] == 101
	assert inputs[0, 1] == 1188
	assert inputs[0, 2] == 1110
	assert inputs[0, 3] == 170
	assert inputs[0, 4] == 170
	print("All tests passed.")
