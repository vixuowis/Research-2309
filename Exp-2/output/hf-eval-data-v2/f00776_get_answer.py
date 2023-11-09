# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(question: str, context: str) -> str:
    """
    This function uses the Hugging Face Transformers library to answer questions based on a given context.

    Args:
        question (str): The question to be answered.
        context (str): The context in which to find the answer.

    Returns:
        str: The answer to the question.
    """
    qa_pipeline = pipeline('question-answering', model='deepset/roberta-large-squad2')
    question_context = {'question': question, 'context': context}
    answer = qa_pipeline(question_context)
    return answer['answer']

# test_function_code --------------------

def test_get_answer():
    """
    This function tests the get_answer function.
    It uses a known question and context to verify that the function returns the correct answer.
    """
    question = 'What is the capital of Germany?'
    context = 'Berlin is the capital of Germany.'
    expected_answer = 'Berlin'
    assert get_answer(question, context) == expected_answer

# call_test_function_code --------------------

test_get_answer()