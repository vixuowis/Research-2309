# requirements_file --------------------

!pip install -U transformers onnx

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_question(context, question):
    """
    Use a pre-trained question answering model to provide answers based on the context.

    Args:
        context (str): The context text containing information to answer the question.
        question (str): The question that needs to be answered based on the context.

    Returns:
        dict: A dictionary containing the answer and additional information.

    Raises:
        ValueError: If the context or the question is empty.
    """
    if not context or not question:
        raise ValueError('The context and question must not be empty.')
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    return qa_pipeline({'context': context, 'question': question})

# test_function_code --------------------

def test_answer_question():
    print("Testing started.")
    context = 'The Transformers library provides state-of-the-art machine learning architectures.'
    question = 'What does the Transformers library provide?'

    # Test case 1: Valid context and question
    print("Testing case [1/2] started.")
    answer = answer_question(context, question)
    assert 'state-of-the-art machine learning architectures' in answer['answer'], f"Test case [1/2] failed: {answer}"

    # Test case 2: Empty context and question
    print("Testing case [2/2] started.")
    try:
        answer_question('', '')
        assert False, "Test case [2/2] failed: No ValueError for empty input."
    except ValueError as e:
        assert str(e) == 'The context and question must not be empty.', f"Test case [2/2] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_question()