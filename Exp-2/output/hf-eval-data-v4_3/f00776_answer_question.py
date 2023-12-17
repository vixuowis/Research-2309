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
    Answers a question based on a given text context using a pre-trained RoBERTa model.

    Args:
        question (str): The question to be answered.
        context (str): The text context containing the information to answer the question.

    Returns:
        dict: Contains the answer and additional information provided by the model.

    Raises:
        ValueError: If the question or context is empty.
    """
    if not question or not context:
        raise ValueError('The question or context cannot be empty.')

    qa_pipeline = pipeline('question-answering', model='deepset/roberta-large-squad2')
    return qa_pipeline({'question': question, 'context': context})

# test_function_code --------------------

def test_answer_question():
    print("Testing started.")

    # Test case 1: Expected to return an answer for a valid question and context
    print("Testing case [1/3] started.")
    context_text = 'Python is an interpreted, high-level, general-purpose programming language.'
    question_text = 'What type of language is Python?'
    answer = answer_question(question_text, context_text)
    assert 'high-level, general-purpose programming language' in answer['answer'], f"Test case [1/3] failed: {answer}"

    # Test case 2: Expecting ValueError for empty question
    print("Testing case [2/3] started.")
    try:
        answer_question('', context_text)
        assert False, "Test case [2/3] failed: ValueError not raised for empty question"
    except ValueError as e:
        assert str(e) == 'The question or context cannot be empty.', f"Test case [2/3] failed: {e}"

    # Test case 3: Expecting ValueError for empty context
    print("Testing case [3/3] started.")
    try:
        answer_question(question_text, '')
        assert False, "Test case [3/3] failed: ValueError not raised for empty context"
    except ValueError as e:
        assert str(e) == 'The question or context cannot be empty.', f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_answer_question()