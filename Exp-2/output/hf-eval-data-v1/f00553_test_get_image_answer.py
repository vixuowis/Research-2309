def test_get_image_answer():
    '''
    This function tests the get_image_answer function with a sample image and question.
    '''
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    question = 'How many people are in this photo?'
    
    # Call the function with the sample image and question
    answer = get_image_answer(url, question)
    
    # Check if the function returns a string (the answer should be a string)
    assert isinstance(answer, str), 'The function should return a string.'
    
    # Print the test result
    print(f'Test passed. The function correctly answered the question: {answer}')

# Run the test function
test_get_image_answer()