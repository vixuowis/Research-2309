# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_question(question, context):
    """
    Answers a question based on a given context using a pretrained BERT model.

    Args:
        question (str): The question to be answered.
        context (str): The context within which to search for the answer.

    Returns:
        str: The answer extracted from the context.

    Raises:
        ValueError: If the `question` or `context` is empty.
    """
    if not question or not context:
        raise ValueError("The `question` and `context` should not be empty.")

    # Initialize the question-answering model
    qa_model = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')

    # Use the model to find the answer
    result = qa_model({'question': question, 'context': context})
    return result['answer']

# test_function_code --------------------

def test_answer_question():
    print("Testing started.")

    # Test case 1: Valid question and context
    print("Testing case [1/2] started.")
    question = 'What is the capital of Sweden?'
    context = 'Stockholm is the beautiful capital of Sweden, which is known for its high living standards and great attractions.'
    expected_answer = 'Stockholm'
    assert answer_question(question, context) == expected_answer, f"Test case [1/2] failed: expected {expected_answer}, got {answer_question(question, context)}"

    # Test case 2: Empty question string
    print("Testing case [2/2] started.")
    question = ''
    context = 'Stockholm is the beautiful capital of Sweden, which is known for its high living standards and great attractions.'
    try:
        answer_question(question, context)
        assert False, "Test case [2/2] failed: ValueError expected but not raised"
    except ValueError as e:
        assert str(e) == "The `question` and `context` should not be empty.", f"Test case [2/2] failed: wrong error message"

    print("Testing finished.")

# call_test_function_line --------------------

test_answer_question()