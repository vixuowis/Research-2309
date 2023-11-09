# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_medical_info(document_text: str, question: str) -> dict:
    """
    This function uses a pre-trained model to extract answers to a given question from a large medical document.

    Args:
        document_text (str): The large medical document from which to extract information.
        question (str): The question to answer.

    Returns:
        dict: The answer extracted from the document. The dictionary contains the 'score', 'start', 'end', and 'answer'.
    """
    qa_pipeline = pipeline('question-answering', model='bigwiz83/sapbert-from-pubmedbert-squad2')
    answer = qa_pipeline({'context': document_text, 'question': question})
    return answer

# test_function_code --------------------

def test_extract_medical_info():
    """
    This function tests the extract_medical_info function.
    It uses a sample document and question, and checks if the function returns a dictionary.
    """
    document_text = 'This model can be loaded on the Inference API on-demand.'
    question = 'Where can the model be loaded?'
    answer = extract_medical_info(document_text, question)
    assert isinstance(answer, dict), 'The function should return a dictionary.'
    assert 'score' in answer, 'The answer should contain a score.'
    assert 'start' in answer, 'The answer should contain a start index.'
    assert 'end' in answer, 'The answer should contain an end index.'
    assert 'answer' in answer, 'The answer should contain the extracted answer.'

# call_test_function_code --------------------

test_extract_medical_info()