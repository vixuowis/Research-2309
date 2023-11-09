def test_translate_text():
    """
    This function tests the 'translate_text' function with some sample data.
    The function uses the assert statement to compare the function's output with the expected output.
    """
    # Test data
    text = 'Hello, how are you?'
    source_lang = 'en'
    target_lang = 'fr'
    # Expected output
    expected_output = 'Bonjour, comment ça va ?'
    # Get the function's output
    output = translate_text(text, source_lang, target_lang)
    # Compare the function's output with the expected output
    assert output == expected_output, f'Expected: {expected_output}, but got: {output}'

test_translate_text()