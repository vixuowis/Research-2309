def test_transcribe_podcast():
    """
    This function tests the transcribe_podcast function by transcribing a sample podcast file and checking the output.
    """
    # Path to a sample podcast file
    sample_podcast_file_path = 'sample_podcast.wav'
    # Transcribe the sample podcast file
    transcription = transcribe_podcast(sample_podcast_file_path)
    # Check that the transcription is a string
    assert isinstance(transcription, str), 'The transcription should be a string.'
    # Check that the transcription is not empty
    assert len(transcription) > 0, 'The transcription should not be empty.'

test_transcribe_podcast()