# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_legal_info(question: str, context: str) -> str:
    """
    Extracts specific information from a legal document using a question-answering model.

    Args:
        question (str): The question related to the legal document.
        context (str): The text from the legal document.

    Returns:
        str: The answer to the question based on the context.

    Raises:
        ImportError: If the required libraries are not installed.
    """
    try:
        doc_qa = pipeline('question-answering', model='Sayantan1993/layoutlmv2-base-uncased_finetuned_docvqa')
        answer = doc_qa(question=question, context=context)
        return answer
    except ImportError:
        raise ImportError('LayoutLMv2Model requires the detectron2 library but it was not found in your environment.')

# test_function_code --------------------

def test_extract_legal_info():
    """
    Tests the function extract_legal_info.
    """
    question = 'What is the contract termination date?'
    context = 'This contract is valid for a period of two years, commencing on the 1st of January 2020 and terminating on the 31st of December 2021.'
    expected_answer = '31st of December 2021'
    assert extract_legal_info(question, context) == expected_answer

    question = 'Who is the contract between?'
    context = 'This contract is between John Doe and Jane Doe.'
    expected_answer = 'John Doe and Jane Doe'
    assert extract_legal_info(question, context) == expected_answer

    question = 'What is the contract duration?'
    context = 'This contract is valid for a period of two years.'
    expected_answer = 'two years'
    assert extract_legal_info(question, context) == expected_answer

    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_legal_info()