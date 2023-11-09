def test_predict_relationship():
    # Test the function with some example sentence pairs
    assert predict_relationship('A man is eating pizza', 'A man eats something') in ['contradiction', 'entailment', 'neutral']
    assert predict_relationship('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.') in ['contradiction', 'entailment', 'neutral']

test_predict_relationship()