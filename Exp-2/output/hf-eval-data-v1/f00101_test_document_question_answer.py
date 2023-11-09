def test_document_question_answer():
    '''
    This function tests the document_question_answer function.
    It uses a sample image and question, and checks if the function returns a non-empty string as answer.
    '''
    # Define a sample image path and question
    image_path = 'path/to/sample/image.png'
    question = 'Sample question'

    # Call the function with the sample input
    answer = document_question_answer(image_path, question)

    # Check if the function returns a non-empty string
    assert isinstance(answer, str), 'The function should return a string.'
    assert len(answer) > 0, 'The function should return a non-empty string.'

test_document_question_answer()