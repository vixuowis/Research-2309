# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(context: str, question: str) -> str:
    '''
    This function uses the BERT large cased whole word masking finetuned model on SQuAD to answer questions on price inflation.
    
    Args:
    context (str): The context in which the question is based.
    question (str): The question to be answered.
    
    Returns:
    str: The answer to the question based on the context.
    
    '''
    qa_pipeline = pipeline('question-answering', model='bert-large-cased-whole-word-masking-finetuned-squad')
    result = qa_pipeline({'context': context, 'question': question})
    return result['answer']

# test_function_code --------------------

def test_get_answer():
    '''
    This function tests the get_answer function.
    
    '''
    context = 'Inflation is an increase in the general price level of goods and services in an economy over time.'
    question = 'What is inflation?'
    answer = get_answer(context, question)
    assert isinstance(answer, str), 'The answer should be a string.'
    assert answer != '', 'The answer should not be an empty string.'

# call_test_function_code --------------------

test_get_answer()