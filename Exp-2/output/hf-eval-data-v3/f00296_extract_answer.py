# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_answer(context: str, question: str) -> str:
    '''
    Extracts the answer to a given question from a given context using a pre-trained model.

    Args:
        context (str): The context from which to extract the answer.
        question (str): The question for which to find the answer.

    Returns:
        str: The extracted answer.
    '''
    qa_pipeline = pipeline('question-answering', model='bigwiz83/sapbert-from-pubmedbert-squad2')
    answer = qa_pipeline({'context': context, 'question': question})
    return answer['answer']

# test_function_code --------------------

def test_extract_answer():
    '''
    Tests the extract_answer function.
    '''
    context = 'This model can be loaded on the Inference API on-demand.'
    question = 'Where can the model be loaded?'
    assert extract_answer(context, question) == 'on the Inference API on-demand'
    context = 'The sky is blue.'
    question = 'What color is the sky?'
    assert extract_answer(context, question) == 'blue'
    context = 'The capital of France is Paris.'
    question = 'What is the capital of France?'
    assert extract_answer(context, question) == 'Paris'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_answer()