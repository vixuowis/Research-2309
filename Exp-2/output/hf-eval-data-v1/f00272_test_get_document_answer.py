def test_get_document_answer():
    """
    This function tests the get_document_answer function.
    It uses a sample image and question, and checks if the returned answer is as expected.
    """
    # Define the test image URL and question
    test_image_url = 'https://example.com/document_invoice.jpg'
    test_question = 'What is the total amount due?'
    
    # Call the function with the test inputs
    test_answer = get_document_answer(test_image_url, test_question)
    
    # Check if the returned answer is a string (since the exact answer can vary depending on the image)
    assert isinstance(test_answer, str), 'The function should return a string.'

test_get_document_answer()