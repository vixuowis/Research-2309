def test_document_question_answer():
    # Test the document_question_answer function with a sample document and question
    document_content = 'This is a sample document. It contains information about various topics.'
    question = 'What does the document contain?'
    answer = document_question_answer(document_content, question)
    assert isinstance(answer, str), 'The function should return a string.'
    assert 'information about various topics' in answer, 'The function should return the correct answer.'

test_document_question_answer()