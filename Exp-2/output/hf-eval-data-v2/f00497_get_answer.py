# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(question: str, context: str) -> str:
    """
    This function uses a pre-trained Korean Electra model to answer a given question based on the provided context.

    Args:
        question (str): The question to be answered.
        context (str): The context within which to find the answer.

    Returns:
        str: The answer to the question.
    """
    qa_pipeline = pipeline('question-answering', model='monologg/koelectra-small-v2-distilled-korquad-384')
    answer = qa_pipeline(question=question, context=context)['answer']
    return answer

# test_function_code --------------------

def test_get_answer():
    """
    This function tests the 'get_answer' function by using a sample question and context.
    """
    question = '고객 질문'
    context = '고객 지원 맥락'
    answer = get_answer(question, context)
    assert isinstance(answer, str), 'The answer should be a string.'

# call_test_function_code --------------------

test_get_answer()