from f00277_tokenize_text import *
def test_tokenize_text():
	question = "What is the capital of France?"
	text = "The capital of France is Paris."
	expected_output = {
		"input_ids": [[101, 2054, 2003, 1996, 3007, 1997, 2605, 1029]],
		"attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1]],
		"token_type_ids": [[0, 0, 0, 0, 0, 0, 0, 0]]
	}

	output = tokenize_text(question, text)
	assert output == expected_output, f"Expected {expected_output}, but got {output}"

# Add more test cases if needed

# Entry point for running the tests
if __name__ == "__main__":
	test_tokenize_text()
