def test_text_to_speech():
    # Test the text_to_speech function
    text = 'Hello, my dog is cute'
    speaker_id = 7306
    audio_file_path = text_to_speech(text, speaker_id)

    # Check that the audio file was created
    assert os.path.exists(audio_file_path)

    # Load the audio file
    speech, samplerate = sf.read(audio_file_path)

    # Check that the speech is not empty
    assert len(speech) > 0

    # Check that the samplerate is correct
    assert samplerate == 16000

test_text_to_speech()