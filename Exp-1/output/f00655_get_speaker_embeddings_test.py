from f00655_get_speaker_embeddings import *
def test_get_speaker_embeddings():
    processed_example = {"speaker_embeddings": [0.1, 0.2, 0.3, ..., 0.9]}
    expected_embeddings = np.array([0.1, 0.2, 0.3, ..., 0.9])
    assert np.array_equal(get_speaker_embeddings(processed_example), expected_embeddings)

    processed_example = {"speaker_embeddings": [1.0, 2.0, 3.0, ..., 9.0]}
    expected_embeddings = np.array([1.0, 2.0, 3.0, ..., 9.0])
    assert np.array_equal(get_speaker_embeddings(processed_example), expected_embeddings)

    processed_example = {"speaker_embeddings": [0.5, 0.5, 0.5, ..., 0.5]}
    expected_embeddings = np.array([0.5, 0.5, 0.5, ..., 0.5])
    assert np.array_equal(get_speaker_embeddings(processed_example), expected_embeddings)

    processed_example = {"speaker_embeddings": [0.0, 0.0, 0.0, ..., 0.0]}
    expected_embeddings = np.array([0.0, 0.0, 0.0, ..., 0.0])
    assert np.array_equal(get_speaker_embeddings(processed_example), expected_embeddings)

    processed_example = {"speaker_embeddings": [0.2, 0.4, 0.6, ..., 0.8]}
    expected_embeddings = np.array([0.2, 0.4, 0.6, ..., 0.8])
    assert np.array_equal(get_speaker_embeddings(processed_example), expected_embeddings)
