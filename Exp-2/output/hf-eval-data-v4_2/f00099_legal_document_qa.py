# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def legal_document_qa(question, context):
    """
    Answer questions based on the content of legal documents.

    Args:
        question (str): The question to answer based on the document.
        context (str): The content of the legal document to analyze.

    Returns:
        dict: A dictionary containing the answer and additional information.

    Raises:
        ValueError: If 'question' or 'context' is not a string or is empty.
    """
    if not question or not isinstance(question, str):
        raise ValueError("'question' must be a non-empty string.")
    if not context or not isinstance(context, str):
        raise ValueError("'context' must be a non-empty string.")

    doc_qa = pipeline('question-answering', model='Sayantan1993/layoutlmv2-base-uncased_finetuned_docvqa')
    answer = doc_qa(question=question, context=context)
    return answer

# test_function_code --------------------

def test_legal_document_qa():
    print("Testing started.")
    question = 'When does the contract expire?'
    context = 'The agreement shall remain in effect until December 31, 2023, unless terminated earlier in accordance with its terms.'

    # Testing case 1: Valid question and context.
    print("Testing case [1/2] started.")
    answer = legal_document_qa(question, context)
    assert 'December 31, 2023' in answer['answer'], f"Test case [1/2] failed: {answer}"

    # Testing case 2: Invalid question type.
    print("Testing case [2/2] started.")
    try:
        legal_document_qa(123, context)
        assert False, "Test case [2/2] failed: No ValueError for non-string question."
    except ValueError as e:
        assert str(e) == "'question' must be a non-empty string.", f"Test case [2/2] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_legal_document_qa()