# function_import --------------------

from transformers import pipeline

# function_code --------------------

def korean_question_answering(question: str, context: str) -> str:
    '''
    This function uses a pre-trained model to answer questions based on the provided context.

    Args:
        question (str): The question to be answered.
        context (str): The context within which to find the answer.

    Returns:
        str: The answer to the question based on the context.
    '''
    korean_qa = pipeline('question-answering', model='monologg/koelectra-small-v2-distilled-korquad-384')
    answer = korean_qa(question=question, context=context)
    return answer['answer']

# test_function_code --------------------

def test_korean_question_answering():
    '''
    This function tests the korean_question_answering function.
    '''
    question = 'What is the capital of South Korea?'
    context = 'The capital of South Korea is Seoul.'
    assert korean_question_answering(question, context) == 'Seoul'
    question = 'Who is the president of South Korea?'
    context = 'The president of South Korea is Moon Jae-in.'
    assert korean_question_answering(question, context) == 'Moon Jae-in'
    question = 'What is the national flower of South Korea?'
    context = 'The national flower of South Korea is the Rose of Sharon.'
    assert korean_question_answering(question, context) == 'Rose of Sharon'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_korean_question_answering()