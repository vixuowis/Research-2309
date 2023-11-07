from f00273_tokenize_text import *
def test_tokenize_text():
	# Test case 1
	question = "What is the capital of France?"
	context = "Paris is the capital of France."
	expected_output = {
		"input_ids": [101, 2054, 2003, 1996, 3007, 1997, 2605, 102],
		"attention_mask": [1, 1, 1, 1, 1, 1, 1, 1]
	}
	assert tokenize_text(question, context) == expected_output

	# Test case 2
	question = "Who wrote the Harry Potter books?"
	context = "J.K. Rowling wrote the Harry Potter books."
	expected_output = {
		"input_ids": [101, 2040, 2356, 1996, 4183, 10759, 2338, 102],
		"attention_mask": [1, 1, 1, 1, 1, 1, 1, 1]
	}
	assert tokenize_text(question, context) == expected_output

	# Test case 3
	question = "What is the capital of Germany?"
	context = "Berlin is the capital of Germany."
	expected_output = {
		"input_ids": [101, 2054, 2003, 1996, 3007, 1997, 2307, 1012, 102],
		"attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1]
	}
	assert tokenize_text(question, context) == expected_output

	# Test case 4
	question = "Who is the CEO of Apple?"
	context = "Tim Cook is the CEO of Apple."
	expected_output = {
		"input_ids": [101, 2040, 2003, 1996, 7397, 1997, 6207, 1012, 102],
		"attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1]
	}
	assert tokenize_text(question, context) == expected_output

	# Test case 5
	question = "What is the largest country in the world?"
	context = "Russia is the largest country in the world."
	expected_output = {
		"input_ids": [101, 2054, 2003, 1996, 2922, 2406, 1999, 1996, 2088, 1012, 102],
		"attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	}
	assert tokenize_text(question, context) == expected_output

