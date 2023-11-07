from f00247_tokenize_text import *
def test_tokenize_text():
	text = "This is a sample text."
	inputs = tokenize_text(text)
	assert inputs is not None

	text = "Another sample text."
	inputs = tokenize_text(text)
	assert inputs is not None

	text = "Yet another sample text."
	inputs = tokenize_text(text)
	assert inputs is not None

	text = "One more sample text."
	inputs = tokenize_text(text)
	assert inputs is not None

	text = "Final sample text."
	inputs = tokenize_text(text)
	assert inputs is not None


test_tokenize_text()
