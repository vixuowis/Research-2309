from f00839_subword_tokenization import *
def test_subword_tokenization():
	text = "I have a new GPU!"
	expected_tokens = ["i", "have", "a", "new", "gp", "##u", "!"]
	assert subword_tokenization(text) == expected_tokens

	text = "This is a test."
	expected_tokens = ["this", "is", "a", "test", "."]
	assert subword_tokenization(text) == expected_tokens

	text = "Hello, world!"
	expected_tokens = ["hello", ",", "world", "!"]
	assert subword_tokenization(text) == expected_tokens

	text = "I am happy."
	expected_tokens = ["i", "am", "happy", "."]
	assert subword_tokenization(text) == expected_tokens

	text = "She is running."
	expected_tokens = ["she", "is", "running", "."]
	assert subword_tokenization(text) == expected_tokens

	print("All test cases pass.")
