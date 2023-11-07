from f00693_generate_python_code import *
def test_generate_python_code():
	model = GPT2LMHeadModel.from_pretrained("gpt2")
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	encoded_zh = {"input_ids": [1, 2, 3, 4, 5]}
	generated_code = generate_python_code(model, tokenizer, encoded_zh)
	expected_code = "Do not interfere with the matters of the witches, because they are delicate and will soon be angry."
	assert generated_code == expected_code


test_generate_python_code()
