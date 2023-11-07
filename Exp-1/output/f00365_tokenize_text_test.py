from f00365_tokenize_text import *
def test_tokenize_text():
	assert tokenize_text("Hello, world!") == torch.tensor([[101, 7592, 1010, 2088, 999, 102]])
	assert tokenize_text("This is a test.") == torch.tensor([[101, 2023, 2003, 1037, 3231, 1012, 102]])
	assert tokenize_text("I love transformers.") == torch.tensor([[101, 1045, 2293, 16404, 1012, 102]])

	print("All test cases pass.")
