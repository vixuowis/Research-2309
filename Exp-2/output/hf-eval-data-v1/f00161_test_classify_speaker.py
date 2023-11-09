def test_classify_speaker():
    '''
    This function tests the classify_speaker function by using a sample audio file.
    '''
    # Define the path to the sample audio file
    sample_audio_file_path = 'tests/samples/ASR/spk1_snt1.wav'
    # Call the classify_speaker function with the sample audio file
    embeddings = classify_speaker(sample_audio_file_path)
    # Assert that the embeddings are not None
    assert embeddings is not None
    # Assert that the embeddings are of the correct type
    assert isinstance(embeddings, torch.Tensor)

test_classify_speaker()