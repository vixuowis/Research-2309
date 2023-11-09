# function_import --------------------

from transformers import pipeline

# function_code --------------------

def korean_question_answering(question: str, context: str) -> str:
    """
    This function uses a pre-trained model to answer questions based on a given context in Korean.

    Args:
        question (str): The question to be answered.
        context (str): The context within which to find the answer.

    Returns:
        str: The answer to the question based on the context.
    """
    korean_qa = pipeline('question-answering', model='monologg/koelectra-small-v2-distilled-korquad-384')
    answer = korean_qa(question=question, context=context)
    return answer['answer']

# test_function_code --------------------

def test_korean_question_answering():
    """
    This function tests the 'korean_question_answering' function with a sample question and context.
    """
    question = '서울의 수도는 무엇인가요?'
    context = '서울은 대한민국의 수도이다.'
    assert korean_question_answering(question, context) == '대한민국'

# call_test_function_code --------------------

test_korean_question_answering()