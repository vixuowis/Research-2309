# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def translate_english_to_german(input_text: str) -> str:
    """
    Translates English text to German using the T5ForConditionalGeneration model from Hugging Face Transformers.

    Args:
        input_text (str): The English text to be translated.

    Returns:
        str: The translated German text.
    """

    # Define a tokenizer and model for T5.
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    t5model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
    
    # Define the input text as a list of strings with each string representing a sentence, since this is what the model expects.
    english_sentences = [input_text]

    # Encode the sentences into IDs using the tokenizer.
    input_ids = tokenizer(english_sentences, padding=True, return_tensors="pt").input_ids
    
    # Generate the output text by feeding it back to the model.
    t5model.config.update({"max_length": 2048})
    generated_ids = t5model.generate(input_ids, decoder_start_token_id=3)
    
    # Tokenize the output and decode it with the tokenizer to return a string.
    translated_texts = [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids] 

    return translated_texts[0]

# test_function_code --------------------

def test_translate_english_to_german():
    assert translate_english_to_german('Where are the parks in Munich?') != ''
    assert translate_english_to_german('How old are you?') != ''
    assert translate_english_to_german('What is your name?') != ''
    return 'All Tests Passed'


# call_test_function_code --------------------

test_translate_english_to_german()