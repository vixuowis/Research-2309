def test_generate_audio_from_text():
    """
    This function tests the 'generate_audio_from_text' function.
    It uses a sample Chinese text and checks if the output is an instance of IPython.lib.display.Audio.
    """
    sample_text = '你好，欢迎来到数字世界。'
    output = generate_audio_from_text(sample_text)
    assert isinstance(output, ipd.lib.display.Audio), 'Output should be an audio file.'

test_generate_audio_from_text()