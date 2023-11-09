# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(context: str, question: str) -> str:
    """
    This function uses a pre-trained model to answer questions based on a given context.

    Args:
        context (str): The context in which the answer is to be found.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question based on the context.
    """
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    result = qa_pipeline({'context': context, 'question': question})
    return result['answer']

# test_function_code --------------------

def test_get_answer():
    """
    This function tests the get_answer function.
    """
    context = 'In 1492, Christopher Columbus sailed the ocean blue, discovering the New World.'
    question = 'Who discovered the New World?'
    answer = get_answer(context, question)
    assert isinstance(answer, str), 'The result should be a string.'
    assert answer.lower() == 'christopher columbus', 'The answer is incorrect.'

# call_test_function_code --------------------

test_get_answer()