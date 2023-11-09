def test_predict_next_word():
    """
    This function tests the predict_next_word function.
    It uses a set of test phrases and checks if the predicted next word is a string.
    """
    test_phrases = [
        'The dog jumped over the',
        'She went to the',
        'I am going to',
        'He loves to play',
        'They are going to the'
    ]
    
    for phrase in test_phrases:
        # Use the function to predict the next word
        predicted_word = predict_next_word(phrase)
        
        # Check if the predicted word is a string
        assert isinstance(predicted_word, str), 'The predicted word is not a string.'

test_predict_next_word()