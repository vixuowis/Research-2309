def test_translate_french_to_english():
    """
    Tests the translate_french_to_english function by translating a French sentence and checking if the output is a string.
    """
    french_text = 'Bonjour, comment ça va?'
    english_text = translate_french_to_english(french_text)
    assert isinstance(english_text, str), 'The output must be a string'

test_translate_french_to_english()