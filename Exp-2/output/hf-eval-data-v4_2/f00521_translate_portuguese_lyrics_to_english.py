# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import MarianMTModel, MarianTokenizer

# function_code --------------------

def translate_portuguese_lyrics_to_english(src_lyrics: str) -> str:
    """
    Translate a string containing lyrics from Portuguese to English.

    Args:
        src_lyrics (str): The lyrics in Portuguese language to be translated.

    Returns:
        str: The translated lyrics in English.

    Raises:
        Exception: If translation fails.

    """
    model_name = 'Helsinki-NLP/opus-mt-pt-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    batch = tokenizer.prepare_translation_batch(src_texts=[src_lyrics])
    output = model.generate(**batch)
    translated_lyrics = tokenizer.decode(output[0], skip_special_tokens=True)
    return translated_lyrics

# test_function_code --------------------

def test_translate_portuguese_lyrics_to_english():
    print("Testing started.")
    # Test case 1: Valid Portuguese lyrics
    print("Testing case [1/1] started.")
    test_lyrics = 'Tudo o que quer de mim irracional é'
    expected_translation = 'All that you want from me is irrational'
    translated_lyrics = translate_portuguese_lyrics_to_english(test_lyrics)
    assert translated_lyrics == expected_translation, f"Test case [1/1] failed: expected {expected_translation}, got {translated_lyrics}"
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_portuguese_lyrics_to_english()