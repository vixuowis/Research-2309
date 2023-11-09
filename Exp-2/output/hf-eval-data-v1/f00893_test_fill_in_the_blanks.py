def test_fill_in_the_blanks():
    sentence = 'The cat chased the [MASK] around the house.'
    filled_sentence = fill_in_the_blanks(sentence)
    assert '[MASK]' not in filled_sentence, 'The function did not fill in the blanks correctly.'
    print('Test passed.')

test_fill_in_the_blanks()