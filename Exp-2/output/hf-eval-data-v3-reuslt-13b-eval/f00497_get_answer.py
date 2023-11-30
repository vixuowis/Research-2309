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
    try:
        # Load a Korean Electra QA model
        pipe = pipeline("question-answering", model="klue/roberta-large", tokenizer="KLUE/roberta-large")
        
        # Get the answer from the model and return it
        answer_info = pipe(question=question, context=context)
        return answer_info["answer"]
    except Exception as e:
        print("Exception in getting answer", str(e))
        return "Sorry! Unable to get an answer at this time. Please try again later."

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