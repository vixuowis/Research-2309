def test_convert_text_to_speech():
    """
    This function tests the 'convert_text_to_speech' function by providing a sample text and checking the type of the output.
    """
    text = 'Bonjour, ceci est un test.'
    wav, rate = convert_text_to_speech(text)
    assert isinstance(wav, np.ndarray), 'The output wav should be a numpy array.'
    assert isinstance(rate, int), 'The output rate should be an integer.'

test_convert_text_to_speech()