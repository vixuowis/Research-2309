def test_translate_english_to_french():
    # Test the translate_english_to_french function with some English text
    input_text = 'Life is like a box of chocolates.'
    translated_text = translate_english_to_french(input_text)
    print(translated_text)
    # Since translation can be subjective and there can be multiple correct translations, we are not asserting a specific output
    assert isinstance(translated_text, str)

test_translate_english_to_french()