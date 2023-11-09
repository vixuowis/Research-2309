def test_correct_grammar():
    """
    This function tests the correct_grammar function with some sample texts.
    """
    test_text1 = 'i can has cheezburger'
    test_text2 = 'the quick brown fox jumps over the laze dog'
    assert correct_grammar(test_text1) == 'I can have a cheeseburger.'
    assert correct_grammar(test_text2) == 'The quick brown fox jumps over the lazy dog.'

test_correct_grammar()