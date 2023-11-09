def test_calculate_sentence_similarity():
    # Test dataset
    sentences = ['This is an example sentence.', 'Each sentence is converted.', 'Calculate the similarity between sentences.']

    # Calculate similarity scores
    similarity_scores = calculate_sentence_similarity(sentences)

    # Check the shape of the output
    assert similarity_scores.shape == (len(sentences), len(sentences)), 'Output shape is incorrect'

    # Check the type of the output
    assert isinstance(similarity_scores, np.ndarray), 'Output type is incorrect'

    # Check the values of the output
    assert np.all(similarity_scores >= 0) and np.all(similarity_scores <= 1), 'Output values are out of range'

test_calculate_sentence_similarity()