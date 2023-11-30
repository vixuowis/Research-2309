# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer_from_text(question: str, context: str) -> str:
    '''
    This function uses a pre-trained model from the transformers library to answer questions based on a given context.

    Args:
        question (str): The question to be answered.
        context (str): The context from which the answer will be extracted.

    Returns:
        str: The answer to the question based on the context.
    '''

    nlp = pipeline('question-answering')
    result = nlp({'context': context, 'question': question})
    
    return result['answer']

# test_function_code --------------------

def test_get_answer_from_text():
    '''
    This function tests the get_answer_from_text function.
    '''
    context = 'Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task.'
    question = 'What is a good example of a question answering dataset?'
    assert get_answer_from_text(question, context) == 'SQuAD dataset'
    question = 'What is Extractive Question Answering?'
    assert get_answer_from_text(question, context) == 'the task of extracting an answer from a text given a question'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_get_answer_from_text()