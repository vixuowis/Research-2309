def test_calculate_sentence_similarity():
    sentence1 = 'This is an example sentence'
    sentence2 = 'This is a similar example sentence'
    sentence3 = 'This is a completely different sentence'

    similarity1 = calculate_sentence_similarity(sentence1, sentence2)
    similarity2 = calculate_sentence_similarity(sentence1, sentence3)

    assert 0 <= similarity1 <= 1, 'Similarity score should be between 0 and 1'
    assert 0 <= similarity2 <= 1, 'Similarity score should be between 0 and 1'
    assert similarity1 > similarity2, 'Similar sentences should have higher similarity scores than dissimilar sentences'