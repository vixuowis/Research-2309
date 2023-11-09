def test_speaker_diarization():
    """
    This function tests the speaker_diarization function.
    It uses a sample audio file for testing.
    """
    # Define the path to the sample audio file and the access token
    audio_file_path = 'path/to/sample/audio.wav'
    access_token = 'SAMPLE_ACCESS_TOKEN'
    
    # Call the speaker_diarization function
    speaker_diarization(audio_file_path, access_token)
    
    # Check if the RTTM file has been created
    assert os.path.exists('audio.rttm')
    
    # Load the RTTM file and check its contents
    with open('audio.rttm', 'r') as rttm:
        content = rttm.read()
    assert len(content) > 0