# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(context: str, question: str) -> str:
    """
    This function uses the Hugging Face transformers library to create a question answering system.
    It uses the 'philschmid/distilbert-onnx' model which is pretrained on the SQuAD dataset and is specifically designed for question answering tasks.

    Args:
        context (str): The text where you are searching for the answer.
        question (str): The user's inquiry.

    Returns:
        str: The best prediction from the model, which can be used to respond to the customer's question.
    """
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    answer = qa_pipeline({'context': context, 'question': question})
    return answer['answer']

# test_function_code --------------------

def test_get_answer():
    context = 'This is a context'
    question = 'What is this?'
    assert get_answer(context, question) == 'a context'
    context = 'The sky is blue'
    question = 'What color is the sky?'
    assert get_answer(context, question) == 'blue'
    context = 'The dog is barking'
    question = 'What is the dog doing?'
    assert get_answer(context, question) == 'barking'
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_get_answer())