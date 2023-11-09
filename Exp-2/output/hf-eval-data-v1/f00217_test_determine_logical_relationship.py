def test_determine_logical_relationship():
    # Test the function with some example sentences
    text1 = 'The cat is on the mat.'
    text2 = 'There is a cat on the mat.'
    result = determine_logical_relationship(text1, text2)
    # Check that the result is a dictionary
    assert isinstance(result, dict)
    # Check that the dictionary contains the keys 'entailment', 'contradiction', and 'neutral'
    assert set(result.keys()) == {'entailment', 'contradiction', 'neutral'}
    # Check that the values in the dictionary are probabilities (i.e., between 0 and 1)
    assert all(0 <= v <= 1 for v in result.values())

test_determine_logical_relationship()