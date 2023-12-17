# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def translate_swedish_to_english(text: str) -> str:
    """
    Translate a given Swedish text into English using the Helsinki-NLP opus-mt-sv-en model.

    Args:
        text (str): The Swedish text to be translated.

    Returns:
        str: The translated English text.
    """
    model_name = 'Helsinki-NLP/opus-mt-sv-en'
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_text = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded_text)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# test_function_code --------------------

def test_translate_swedish_to_english():
    print("Testing started.")
    # Testing a known translation example
    swedish_text = 'Stockholm är Sveriges huvudstad och största stad. Den har en rik historia och erbjuder många kulturella och historiska sevärdheter.'
    expected_translation = 'Stockholm is the capital and largest city of Sweden. It has a rich history and offers many cultural and historical attractions.'

    # Test case 1
    print("Testing case [1/1] started.")
    english_text = translate_swedish_to_english(swedish_text)
    assert english_text == expected_translation, f"Test case [1/1] failed: Expected '{{expected_translation}}', got '{{english_text}}'"
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_swedish_to_english()