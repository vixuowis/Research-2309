def test_sentence_similarity():
    # Test the sentence_similarity function with some example sentences
    sentence1 = 'This is the first sentence.'
    sentence2 = 'This is the second sentence.'
    sentence3 = 'This is an entirely different sentence.'
    
    # The similarity between sentence1 and sentence2 should be high
    assert sentence_similarity(sentence1, sentence2) > 0.8
    
    # The similarity between sentence1 and sentence3 should be low
    assert sentence_similarity(sentence1, sentence3) < 0.5
    
    print('All tests passed.')

test_sentence_similarity()