def test_generate_summary_and_question():
    """
    This function tests the 'generate_summary_and_question' function by using a sample text.
    It asserts that the output is not None, indicating that the function has successfully generated a summary and open-ended question.
    """
    # Define a sample text
    sample_text = "This is a sample text for testing the function."
    
    # Call the function with the sample text
    output = generate_summary_and_question(sample_text)
    
    # Assert that the output is not None
    assert output is not None, "Test failed! The function did not return a result."