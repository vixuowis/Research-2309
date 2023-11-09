def test_analyze_sentiment():
    """
    This function tests the analyze_sentiment function with some example texts.
    """
    # Positive sentiment text
    text1 = 'I love the new product!'
    result1 = analyze_sentiment(text1)
    assert result1[0]['label'] == 'POSITIVE', 'Test Case 1 Failed'
    
    # Negative sentiment text
    text2 = 'I hate the new product!'
    result2 = analyze_sentiment(text2)
    assert result2[0]['label'] == 'NEGATIVE', 'Test Case 2 Failed'
    
    print('All test cases pass')

test_analyze_sentiment()