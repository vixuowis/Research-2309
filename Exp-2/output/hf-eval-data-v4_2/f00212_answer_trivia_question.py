# requirements_file --------------------

!pip install -U transformers onnx

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_trivia_question(context: str, question: str) -> str:
    """
    Answers trivia questions about history using a pre-trained model.

    Args:
        context (str): The context text containing information relevant to the question.
        question (str): The trivia question to be answered.

    Returns:
        str: The answer to the trivia question derived from the context.

    Raises:
        ValueError: If the context or question is empty.
    """
    if not context or not question:
        raise ValueError('The context and question must be non-empty strings.')
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    return qa_pipeline({'context': context, 'question': question})['answer']

# test_function_code --------------------

def test_answer_trivia_question():
    print("Testing started.")

    context = 'In 1492, Christopher Columbus sailed the ocean blue, discovering the New World.'
    question = 'Who discovered the New World?'
    expected_answer = 'Christopher Columbus'

    print("Testing case [1/1] started.")
    actual_answer = answer_trivia_question(context, question)
    assert actual_answer == expected_answer, f"Test case [1/1] failed: expected {expected_answer}, got {actual_answer}"

    print("Testing finished.")

# call_test_function_line --------------------

test_answer_trivia_question()