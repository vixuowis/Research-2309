def test_analyze_image():
    """
    This function tests the 'analyze_image' function.
    It uses a sample image and question, and checks if the function returns a result.
    """
    # Define a sample image path and question
    image_path = 'path/to/sample/image.jpg'
    question = 'What is in the image?'
    
    # Call the 'analyze_image' function
    result = analyze_image(image_path, question)
    
    # Check if the function returns a result
    assert result is not None, 'The function did not return a result.'
    
    # Print the result for manual inspection
    print(f'Question: {question}')
    print(f'Answer: {result}')

test_analyze_image()