# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# function_code --------------------

def translate_spanish_to_polish(spanish_text):
    """
    Translates Spanish text to Polish using the MBart50 model.

    Args:
        spanish_text (str): The Spanish text to be translated.

    Returns:
        str: The translated text in Polish.
    """

    # Load the MBart model and tokenizer
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer.src_lang = 'es_ES'

    # Tokenize the Spanish text for the model
    encoded_spanish = tokenizer(spanish_text, return_tensors='pt')

    # Translate the text to Polish
    generated_tokens = model.generate(
        **encoded_spanish,
        forced_bos_token_id=tokenizer.lang_code_to_id['pl_PL']
    )

    # Decode the tokens to a string
    polish_subtitles = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return polish_subtitles

# test_function_code --------------------

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def test_translate_spanish_to_polish():
    print("Testing started.")
    # Mock Spanish text to test the function
    spanish_text = 'Hola mundo'

    # Expected Polish translation for 'Hola mundo'
    expected_polish = 'Witaj świecie'

    # Testing case 1
    print("Testing case [1/1] started.")
    translated_text = translate_spanish_to_polish(spanish_text)
    assert translated_text == expected_polish, f"Test case [1/1] failed: Expected '{{expected_polish}}', got '{{translated_text}}'"
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_spanish_to_polish()