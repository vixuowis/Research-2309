def test_extract_medical_info():
    '''
    This function tests the extract_medical_info function.
    '''
    # Define a sample document and question
    document_text = 'This model can be loaded on the Inference API on-demand.'
    question = 'Where can the model be loaded?'

    # Call the function with the sample document and question
    answer = extract_medical_info(document_text, question)

    # Assert that the function returns the expected answer
    assert 'Inference API' in answer['answer'], 'Test failed!'

test_extract_medical_info()