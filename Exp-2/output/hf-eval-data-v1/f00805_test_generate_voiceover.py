def test_generate_voiceover():
    """
    Test the function generate_voiceover.
    """
    # Test with a sample text
    generate_voiceover('Mary hatte ein kleines Lamm')
    # Check if the audio file is generated
    assert os.path.exists('example_TTS.wav')