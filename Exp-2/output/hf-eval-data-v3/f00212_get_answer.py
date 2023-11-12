# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(context: str, question: str) -> str:
    """
    This function uses the Hugging Face's transformers library to answer questions based on a given context.

    Args:
        context (str): The context in which the answer to the question is contained.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question.

    Raises:
        ValueError: If the model cannot be loaded.
    """
    try:
        qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
        answer = qa_pipeline({'context': context, 'question': question})
        return answer['answer']
    except Exception as e:
        raise ValueError('Could not load model with the specified classes.') from e

# test_function_code --------------------

def test_get_answer():
    """
    This function tests the get_answer function.
    """
    context = 'In 1492, Christopher Columbus sailed the ocean blue, discovering the New World.'
    question = 'Who discovered the New World?'
    answer = get_answer(context, question)
    assert answer == 'Christopher Columbus', f'Error: {answer}'

    context = 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.'
    question = 'Where is the Eiffel Tower located?'
    answer = get_answer(context, question)
    assert answer == 'Paris, France', f'Error: {answer}'

    context = 'Python is an interpreted, high-level, general-purpose programming language.'
    question = 'What is Python?'
    answer = get_answer(context, question)
    assert answer == 'an interpreted, high-level, general-purpose programming language', f'Error: {answer}'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_answer()