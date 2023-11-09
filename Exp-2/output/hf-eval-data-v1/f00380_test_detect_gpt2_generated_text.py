def test_detect_gpt2_generated_text():
    """
    This function tests the 'detect_gpt2_generated_text' function by using a sample text.
    """
    # Define a sample text
    sample_text = 'Hello world! Is this content AI-generated?'
    
    # Call the 'detect_gpt2_generated_text' function with the sample text
    prediction = detect_gpt2_generated_text(sample_text)
    
    # Assert that the function returns a dictionary (the prediction result)
    assert isinstance(prediction, dict)
    
    # Assert that the prediction result contains the 'label' and 'score' keys
    assert 'label' in prediction and 'score' in prediction

test_detect_gpt2_generated_text()