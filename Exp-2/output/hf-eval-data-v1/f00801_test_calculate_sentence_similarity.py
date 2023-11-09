def test_calculate_sentence_similarity():
    sentence1 = 'This is a test sentence.'
    sentence2 = 'This is a test sentence.'
    sentence3 = 'This is a completely different sentence.'

    assert np.isclose(calculate_sentence_similarity(sentence1, sentence2), 1, atol=1e-5), 'Test Case 1 Failed'
    assert np.isclose(calculate_sentence_similarity(sentence1, sentence3), 0, atol=1e-5), 'Test Case 2 Failed'
    print('All test cases pass')

test_calculate_sentence_similarity()