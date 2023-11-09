# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(question: str, context: str) -> str:
    """
    This function uses the Hugging Face Transformers library to answer a question based on a given context.

    Args:
        question (str): The question to be answered.
        context (str): The context in which to find the answer.

    Returns:
        str: The answer to the question.
    """
    qa_model = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')
    result = qa_model({'question': question, 'context': context})
    return result['answer']

# test_function_code --------------------

def test_get_answer():
    """
    This function tests the get_answer function.
    """
    question = 'What is the capital of Sweden?'
    context = 'Stockholm is the beautiful capital of Sweden, which is known for its high living standards and great attractions.'
    answer = get_answer(question, context)
    assert answer == 'Stockholm', f'Error: {answer}'

# call_test_function_code --------------------

test_get_answer()