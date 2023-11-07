from f00195_preprocess_function import *
def test_preprocess_function():
	test_input = {"text": "This is an example sentence."}
	expected_output = {"input_ids": [101, 2023, 2003, 2019, 2742, 6251, 1012, 102], "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1]}
	assert preprocess_function(test_input) == expected_output
	# Add more test cases here
