from f00291_create_data_collator_for_language_modeling import *
def test_create_data_collator_for_language_modeling():
	tokenizer = Tokenizer()
	data_collator = create_data_collator_for_language_modeling(tokenizer)
	assert isinstance(data_collator, DataCollatorForLanguageModeling)

	# Add test cases here

if __name__ == '__main__':
	test_create_data_collator_for_language_modeling()
