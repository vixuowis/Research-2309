def test_generate_abstract():
    """
    This function tests the 'generate_abstract' function with a sample input text.
    """
    # Define the input text
    input_text = 'Studies have shown the impacts of social media on mental health'
    
    # Call the 'generate_abstract' function with the input text
    abstract = generate_abstract(input_text)
    
    # Assert that the generated abstract is not empty
    assert abstract != '', 'The generated abstract is empty.'
    
    # Print the generated abstract
    print('Generated abstract:', abstract)
    
# Call the test function
test_generate_abstract()