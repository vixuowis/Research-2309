# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def find_answer_in_document(context, question):
    """
    This function uses a pre-trained BERT model to find answers to questions from a given document.

    Parameters:
        context (str): A string containing the document text.
        question (str): The question to find an answer for in the document.

    Returns:
        dict: A dictionary containing the answer and additional information.
    """
    qa_tool = pipeline('question-answering', model='bert-large-cased-whole-word-masking-finetuned-squad')
    answer = qa_tool({'context': context, 'question': question})
    return answer

# test_function_code --------------------

def test_find_answer_in_document():
    print("Testing started.")
    context = 'This is a long document containing company policies, financial details, and team structures.'
    question = 'What are the company policies mentioned in the document?'

    # Test case 1: Check if the function returns a dictionary
    print("Testing case [1/1] started.")
    result = find_answer_in_document(context, question)
    assert isinstance(result, dict), f"Test case [1/1] failed: Expected result to be a dictionary, got {type(result)} instead."
    print("Testing finished.")

# Run the test function
test_find_answer_in_document()