# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer_from_document(question: str, document: str) -> str:
    """
    This function uses the Hugging Face Transformers pipeline to answer questions based on a given document.

    Args:
        question (str): The question to be answered.
        document (str): The document to be used as context for answering the question.

    Returns:
        str: The answer to the question.

    Raises:
        OSError: If there is an issue with disk space when loading the model.
    """
    document_qa_model = pipeline('question-answering', model='pardeepSF/layoutlm-vqa')
    answer = document_qa_model(question=question, context=document)
    return answer['answer']

# test_function_code --------------------

def test_get_answer_from_document():
    """
    This function tests the get_answer_from_document function.
    """
    question = 'What is the cost of a deluxe suite per night?'
    document = 'The cost of a deluxe suite per night is $200.'
    assert get_answer_from_document(question, document) == '$200'
    question = 'What is the check-in time?'
    document = 'The check-in time is 3 PM.'
    assert get_answer_from_document(question, document) == '3 PM'
    question = 'Is breakfast included?'
    document = 'Yes, breakfast is included in the cost.'
    assert get_answer_from_document(question, document) == 'Yes'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_answer_from_document()