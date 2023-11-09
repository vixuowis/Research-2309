def test_calculate_sentence_similarity():
    '''
    This function tests the calculate_sentence_similarity function.
    It uses a set of test sentences and checks if the similarity score is within an acceptable range.
    '''
    sentence1 = 'I love going to the park'
    sentence2 = 'My favorite activity is visiting the park'
    similarity = calculate_sentence_similarity(sentence1, sentence2)
    assert 0.7 <= similarity <= 1.0, 'Test failed!'

test_calculate_sentence_similarity()