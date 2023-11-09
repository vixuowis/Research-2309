def test_fill_mask_french():
    '''
    This function tests the fill_mask_french function.
    It uses a few sample sentences and checks if the function correctly fills in the missing words.
    '''
    # Test sentences
    test_sentences = [
        'Le camembert est <mask> :)',
        'Paris est la <mask> de la France',
        'Je <mask> au supermarché'
    ]
    # Expected results
    expected_results = [
        'Le camembert est délicieux :)',
        'Paris est la capitale de la France',
        'Je vais au supermarché'
    ]
    # Test the function
    for i, sentence in enumerate(test_sentences):
        result = fill_mask_french(sentence)
        # Check if the result is as expected
        assert result == expected_results[i], f'Expected "{expected_results[i]}", but got "{result}"'

test_fill_mask_french()