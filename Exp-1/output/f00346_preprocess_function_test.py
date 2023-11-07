from f00346_preprocess_function import *
source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

def test_preprocess_function():
	examples = {"translation": [{"en": "Hello", "fr": "Bonjour"}, {"en": "Goodbye", "fr": "Au revoir"}]}
	expected_model_inputs = tokenizer([prefix + example[source_lang] for example in examples["translation"]], text_target=[example[target_lang] for example in examples["translation"]], max_length=128, truncation=True)
	assert preprocess_function(examples) == expected_model_inputs
