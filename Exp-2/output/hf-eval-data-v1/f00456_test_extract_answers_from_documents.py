def test_extract_answers_from_documents():
    # Test the function with a sample document and questions
    document = 'The capital of France is Paris. The country is located in Europe and uses the Euro as its currency.'
    questions = ['What is the capital of France?', 'Where is France located?', 'What currency does France use?']
    expected_answers = {'What is the capital of France?': 'Paris', 'Where is France located?': 'Europe', 'What currency does France use?': 'Euro'}
    assert extract_answers_from_documents(questions, document) == expected_answers

test_extract_answers_from_documents()