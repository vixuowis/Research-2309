def test_pos_tagging():
    """
    Function to test the 'pos_tagging' function.
    """
    test_text = 'The quick brown fox jumps over the lazy dog.'
    expected_output = [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN'), ('.', '.')]
    assert pos_tagging(test_text) == expected_output

test_pos_tagging()