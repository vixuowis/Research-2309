# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(context_text: str, question: str) -> str:
    """
    This function uses a pre-trained model to answer questions based on the provided context.

    Args:
        context_text (str): The context in which the question should be answered.
        question (str): The question that needs to be answered.

    Returns:
        str: The answer to the question based on the context.
    """
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    answer = qa_pipeline({'context': context_text, 'question': question})
    return answer['answer']

# test_function_code --------------------

def test_get_answer():
    """
    This function tests the 'get_answer' function with a sample context and question.
    """
    context_text = 'The sky is blue.'
    question = 'What color is the sky?'
    answer = get_answer(context_text, question)
    assert isinstance(answer, str), 'The answer should be a string.'
    assert answer.lower() == 'blue', 'The answer should be blue.'

# call_test_function_code --------------------

test_get_answer()