# requirements_file --------------------

import subprocess

requirements = ["transformers", "onnx"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_question(context, question):
    """
    Answers a question based on the given context using a pretrained Transformer model.

    Args:
        context (str): The passage or document in which to search for answers.
        question (str): The question to be answered.

    Returns:
        str: The predicted answer to the question based on the context.

    Raises:
        ValueError: If context or question is not provided.
    """
    if not context or not question:
        raise ValueError('Both context and question must be provided.')

    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    answer = qa_pipeline({'context': context, 'question': question})
    return answer['answer']

# test_function_code --------------------

def test_answer_question():
    print("Testing started.")

    # Test case 1: Standard question
    context = "This is a sample context where we're testing the model."
    question = "Where are we testing the model?"
    print("Testing case [1/3] started.")
    answer1 = answer_question(context, question)
    assert answer1 == 'sample context', f"Test case [1/3] failed: Expected 'sample context', got {answer1}"

    # Test case 2: Question not in context
    question = "What is the color of the sky?"
    print("Testing case [2/3] started.")
    answer2 = answer_question(context, question)
    assert answer2 == 'No answer found.', f"Test case [2/3] failed: Expected 'No answer found.', got {answer2}"

    # Test case 3: Empty context
    context = ""
    question = "How?"
    print("Testing case [3/3] started.")
    try:
        answer_question(context, question)
    except ValueError as e:
        assert str(e) == 'Both context and question must be provided.', f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_answer_question()