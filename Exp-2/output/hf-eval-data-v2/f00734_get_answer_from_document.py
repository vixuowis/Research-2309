# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer_from_document(question: str, document: str) -> str:
    """
    This function uses the Hugging Face Transformers library to answer questions based on a given document.

    Args:
        question (str): The question that needs to be answered.
        document (str): The document from which the answer needs to be extracted.

    Returns:
        str: The answer to the question based on the document.
    """
    document_qa_model = pipeline('question-answering', model='pardeepSF/layoutlm-vqa')
    answer = document_qa_model(question=question, context=document)
    return answer['answer']

# test_function_code --------------------

def test_get_answer_from_document():
    """
    This function tests the 'get_answer_from_document' function by using a sample question and document.
    """
    question = 'What is the cost of a deluxe suite per night?'
    document = 'The cost of a deluxe suite per night is $200.'
    assert get_answer_from_document(question, document) == '$200'

# call_test_function_code --------------------

test_get_answer_from_document()