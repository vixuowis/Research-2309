# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(question: str, context: str) -> str:
    '''
    This function uses a pre-trained Korean Electra model to answer a given question based on the provided context.

    Args:
        question (str): The question to be answered.
        context (str): The context within which to find the answer.

    Returns:
        str: The answer to the question.
    '''
    
    # Initialize an extractive question-answering pipeline using a pre-trained Korean Electra model
    pipe = pipeline('question-answering', model="monologg/koelectra-small-v3-finetuned-korquad-korean", tokenizer="monologg/koelectra-small-v3-finetuned-korquad-korean")
    
    # Get answer to the question based on the context using the pipeline we initialized
    result = pipe(question=question, context=context)
    
    return result['answer']


# test_function_code --------------------

def test_get_answer():
    '''
    This function tests the get_answer function.
    '''
    question = '고객 질문'
    context = '고객 지원 맥락'
    assert isinstance(get_answer(question, context), str)
    question = '또 다른 고객 질문'
    context = '또 다른 고객 지원 맥락'
    assert isinstance(get_answer(question, context), str)
    question = '세 번째 고객 질문'
    context = '세 번째 고객 지원 맥락'
    assert isinstance(get_answer(question, context), str)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_get_answer()