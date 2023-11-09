# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_legal_info(question: str, context: str) -> str:
    """
    This function uses a pre-trained model from Hugging Face Transformers to answer questions based on a given context.
    The model used is 'Sayantan1993/layoutlmv2-base-uncased_finetuned_docvqa', which is trained for document question answering tasks.

    Args:
        question (str): The question that needs to be answered.
        context (str): The context in which the question is to be answered.

    Returns:
        str: The answer to the question based on the context.
    """
    doc_qa = pipeline('question-answering', model='Sayantan1993/layoutlmv2-base-uncased_finetuned_docvqa')
    answer = doc_qa(question=question, context=context)
    return answer['answer']

# test_function_code --------------------

def test_extract_legal_info():
    """
    This function tests the 'extract_legal_info' function by providing a sample question and context.
    The function asserts if the returned answer is not as expected.
    """
    question = 'What is the contract termination date?'
    context = 'This contract is valid for a period of two years, commencing on the 1st of January 2020 and terminating on the 31st of December 2021.'
    expected_answer = '31st of December 2021'
    assert extract_legal_info(question, context) == expected_answer

# call_test_function_code --------------------

test_extract_legal_info()