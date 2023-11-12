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

    Raises:
        ValueError: If the model cannot be loaded.
    """
    try:
        qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
        answer = qa_pipeline({'context': context_text, 'question': question})
        return answer['answer']
    except Exception as e:
        raise ValueError('Could not load model with the following error: ' + str(e))

# test_function_code --------------------

def test_get_answer():
    """
    This function tests the get_answer function with different test cases.
    """
    context_text = 'This is a context'
    question = 'What is this?'
    assert isinstance(get_answer(context_text, question), str)
    context_text = 'The sky is blue'
    question = 'What color is the sky?'
    assert isinstance(get_answer(context_text, question), str)
    context_text = 'I have a dog'
    question = 'What do I have?'
    assert isinstance(get_answer(context_text, question), str)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_answer()