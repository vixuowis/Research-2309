def test_get_answer_from_document():
    document_text = 'In a healthcare company, we are trying to create an automated system for answering patient-related questions based on their medical documents.'
    question_text = 'What is the company trying to create?'
    answer = get_answer_from_document(document_text, question_text)
    assert isinstance(answer, str), 'The function should return a string.'
    assert 'automated system for answering patient-related questions' in answer, 'The function returned an incorrect answer.'

test_get_answer_from_document()