def test_improve_sentence_readability():
    '''
    This function tests the improve_sentence_readability function.
    It uses a set of test sentences and checks if the function returns
    a sentence without the masked part.
    '''
    test_sentences = [
        'The cat was chasing its [MASK].',
        'I am going to the [MASK] after school.',
        'She is a [MASK] student.',
        'We are planning to visit [MASK] this summer.',
        'He is a [MASK] of the basketball team.'
    ]

    for sentence in test_sentences:
        improved_sentence = improve_sentence_readability(sentence)
        assert '[MASK]' not in improved_sentence, 'The function did not replace the masked part.'

test_improve_sentence_readability()