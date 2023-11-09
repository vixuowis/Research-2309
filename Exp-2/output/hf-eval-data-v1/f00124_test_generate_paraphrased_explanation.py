def test_generate_paraphrased_explanation():
    """
    This function tests the 'generate_paraphrased_explanation' function by using a sample chemistry concept text.
    """
    # Define a sample chemistry concept text
    sample_text = 'A chemical reaction is a process that leads to the chemical transformation of one set of chemical substances to another.'
    
    # Generate a paraphrased explanation for the sample text
    paraphrased_text = generate_paraphrased_explanation(sample_text)
    
    # Assert that the paraphrased text is not the same as the original text
    assert paraphrased_text != sample_text, 'The paraphrased text is the same as the original text.'
    
    # Assert that the paraphrased text is not empty
    assert paraphrased_text, 'The paraphrased text is empty.'
    
    print('All tests passed.')

# Run the test function
test_generate_paraphrased_explanation()