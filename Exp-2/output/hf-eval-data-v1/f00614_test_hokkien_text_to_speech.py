def test_hokkien_text_to_speech():
    """
    This function tests the hokkien_text_to_speech function by providing a sample Hokkien text and checking if the output is of type IPython.lib.display.Audio.
    """
    sample_text = 'Insert Hokkien text here'
    output = hokkien_text_to_speech(sample_text)
    assert isinstance(output, ipd.lib.display.Audio), 'Output should be an audio file'