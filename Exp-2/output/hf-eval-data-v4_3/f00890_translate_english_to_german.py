# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# function_code --------------------

def translate_english_to_german(text: str) -> str:
    """
    Translate English text to German using a pre-trained mBART model.

    Args:
        text (str): The English text to be translated.

    Returns:
        str: The translated German text.

    Raises:
        RuntimeError: If an error occurs during the translation process.
    """
    try:
        model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
        tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50', src_lang='en_XX', tgt_lang='de_DE')
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model.generate(**inputs)
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return translation
    except Exception as e:
        raise RuntimeError(f'An error occurred during translation: {e}')

# test_function_code --------------------

def test_translate_english_to_german():
    print("Testing started.")
    # Test case 1: Simple greeting
    print("Testing case [1/3] started.")
    assert translate_english_to_german("Hello, world!") == 'Hallo, Welt!', "Test case [1/3] failed: Incorrect translation of simple greeting."

    # Test case 2: A common phrase
    print("Testing case [2/3] started.")
    assert translate_english_to_german("Thank you for your cooperation.") == 'Danke für Ihre Zusammenarbeit.', "Test case [2/3] failed: Incorrect translation of a common phrase."

    # Test case 3: Handling of unknown words
    print("Testing case [3/3] started.")
    assert translate_english_to_german("This is a neologism: techmology.") != '', "Test case [3/3] failed: Did not handle unknown words correctly."
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_english_to_german()