# requirements_file --------------------

import subprocess

requirements = ["transformers", "sentencepiece"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# function_code --------------------

def translate_to_french(english_text):
    """
    Translates English text to French using a pre-trained M2M100 model.

    Args:
        english_text (str): The text in English to be translated.

    Returns:
        str: The translated text in French.

    Raises:
        ValueError: If the `english_text` is empty or not provided.
    """
    if not english_text:
        raise ValueError('No English text provided for translation.')

    model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
    tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')
    tokenizer.src_lang = 'en'

    encoded_input = tokenizer(english_text, return_tensors='pt')
    generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id('fr'))

    french_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return french_text

# test_function_code --------------------

def test_translate_to_french():
    print("Testing started.")
    # Normal case
    print("Testing case [1/2] started.")
    sample_text = "Welcome to our hotel, we hope you enjoy your stay."
    translated_text = translate_to_french(sample_text)
    assert translated_text == "Bienvenue à notre hôtel, nous espérons que vous apprécierez votre séjour.", f"Test case [1/2] failed: {translated_text}"

    # Error case
    print("Testing case [2/2] started.")
    try:
        invalid_text = ""
        translate_to_french(invalid_text)
        assert False, "Test case [2/2] failed: No ValueError raised for empty input."
    except ValueError as e:
        assert str(e) == "No English text provided for translation."
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_to_french()