# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def translate_content(text, model='facebook/nllb-200-distilled-600M'):
    """
    Translate the input text into another language using the specified model.

    Parameters:
        text (str): The text to translate.
        model (str): The translation model to use. Defaults to 'facebook/nllb-200-distilled-600M'.

    Returns:
        str: The translated text.
    """
    translator = pipeline('translation_xx_to_yy', model=model)
    translated_text = translator(text)[0]['translation_text']
    return translated_text

# test_function_code --------------------

def test_translate_content():
    print("Testing translate_content function.")
    # Test case 1: Translate English to French
    print("Testing case [1/3] started.")
    translated_text = translate_content('Hello, world!', 'facebook/nllb-200-distilled-600M')
    assert translated_text != 'Hello, world!', "Test case [1/3] failed: Translation should not be the same as input."

    # Test case 2: Translate English to Spanish
    print("Testing case [2/3] started.")
    translated_text = translate_content('Good morning', 'facebook/nllb-200-distilled-600M')
    assert translated_text != 'Good morning', "Test case [2/3] failed: Translation should not be the same as input."

    # Test case 3: Empty string input
    print("Testing case [3/3] started.")
    translated_text = translate_content('', 'facebook/nllb-200-distilled-600M')
    assert translated_text == '', "Test case [3/3] failed: Translation of empty text should be empty."
    print("Testing finished.")

# Run the test function
test_translate_content()