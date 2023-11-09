def test_text_to_speech():
    """
    Tests the text_to_speech function by converting a sample text into speech and checking if the output file is created.
    """
    import os
    sample_text = 'The sun was shining brightly, and the birds were singing sweetly.'
    output_file = 'test_TTS.wav'
    text_to_speech(sample_text, output_file)
    assert os.path.exists(output_file), 'Output file not created.'
    os.remove(output_file)  # clean up after test