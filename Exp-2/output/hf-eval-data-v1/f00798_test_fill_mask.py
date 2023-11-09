def test_fill_mask():
    """
    This function tests the 'fill_mask' function with some sample texts.
    """
    test_texts = [
        'The weather was so <mask> that everyone stayed indoors.',
        'She is the <mask> of the team.',
        'This is a <mask> day.'
    ]
    expected_outputs = [
        'The weather was so bad that everyone stayed indoors.',
        'She is the leader of the team.',
        'This is a great day.'
    ]
    for i, text in enumerate(test_texts):
        assert fill_mask(text) in expected_outputs[i]