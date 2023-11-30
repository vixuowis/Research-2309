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
    # Load the pre-trained tokenizer
    english_german_tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-english-to-german")

    # Prepare the English text for translation: add the task prefix 
    input_text = "translate English to German: " + input_text
    
    # Generate tokenized sequence
    tokenized_text = english_german_tokenizer.encode(input_text, return_tensors="pt")

    # Load the pre-trained T5ForConditionalGeneration model and generate German text using the English text as input
    german_translated_model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-english-to-german")
    translated = german_translated_model.generate(tokenized_text)
    
    # Transform the generated tokenized sequence back to German text
    translated_german_text = english_german_tokenizer.decode(translated[0], skip_special_tokens=True)    

    return translated_german_text

# test_function_code --------------------

def test_translate_english_to_german():
    assert translate_english_to_german('Where are the parks in Munich?') != ''
    assert translate_english_to_german('How old are you?') != ''
    assert translate_english_to_german('What is your name?') != ''
    return 'All Tests Passed'


# call_test_function_code --------------------

test_translate_english_to_german()