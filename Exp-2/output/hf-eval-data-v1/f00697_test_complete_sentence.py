def test_complete_sentence():
    '''
    This function tests the 'complete_sentence' function with some test cases.
    '''
    assert 'dark' in complete_sentence('In the story, the antagonist represents the <mask> nature of humanity.')
    assert 'complex' in complete_sentence('The <mask> nature of the problem made it difficult to solve.')
    assert 'human' in complete_sentence('The <mask> spirit is indomitable.')

test_complete_sentence()