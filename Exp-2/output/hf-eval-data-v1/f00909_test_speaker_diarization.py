def test_speaker_diarization():
    """
    Test function for speaker_diarization function.
    """
    # Assuming there is a test audio file named 'test_audio.wav' in the current directory
    speaker_diarization('test_audio.wav', 'ACCESS_TOKEN_GOES_HERE')
    # Check if the output file is created
    assert os.path.exists('test_audio.wav.rttm')
    # Check if the output file is not empty
    assert os.path.getsize('test_audio.wav.rttm') > 0