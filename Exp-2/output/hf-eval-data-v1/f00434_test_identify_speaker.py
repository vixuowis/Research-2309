def test_identify_speaker():
    # Test the identify_speaker function with a sample audio file
    # Note: Replace 'sample.wav' with the path to your test audio file
    embeddings = identify_speaker('sample.wav')

    # Check that the function returns an array
    assert isinstance(embeddings, np.ndarray), 'The function should return a numpy array.'

    # Check that the array is not empty
    assert embeddings.size > 0, 'The function should return a non-empty array.'

    # Check that the array has the correct shape
    assert embeddings.shape[1] == 768, 'The function should return an array with shape (n, 768), where n is the number of frames in the audio file.'

test_identify_speaker()