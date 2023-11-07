from f00669_generate_audio import *
def test_generate_audio():
    text = "Hello, how are you?"
    speaker_embeddings = np.random.randn(128)

    output = generate_audio(text, speaker_embeddings)

    assert 'audio' in output
    assert 'sampling_rate' in output
    assert isinstance(output['audio'], np.ndarray)
    assert isinstance(output['sampling_rate'], int)

    print('All tests passed!')


test_generate_audio()
