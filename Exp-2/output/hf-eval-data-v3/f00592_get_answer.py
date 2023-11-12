# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(context: str, question: str) -> str:
    """
    This function uses the BERT large cased whole word masking finetuned model on SQuAD to answer questions on price inflation.

    Args:
        context (str): The context in which the question is asked.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question.

    Raises:
        OSError: If there is a problem with the disk quota.
    """
    qa_pipeline = pipeline('question-answering', model='bert-large-cased-whole-word-masking-finetuned-squad')
    result = qa_pipeline({'context': context, 'question': question})
    return result['answer']

# test_function_code --------------------

def test_get_answer():
    context = 'Inflation is an increase in the general price level of goods and services in an economy over time.'
    question = 'What is inflation?'
    answer = get_answer(context, question)
    assert isinstance(answer, str)
    assert answer == 'an increase in the general price level of goods and services in an economy over time.'

    context = 'The rate of inflation is the change in prices for goods and services over time.'
    question = 'What is the rate of inflation?'
    answer = get_answer(context, question)
    assert isinstance(answer, str)
    assert answer == 'the change in prices for goods and services over time.'

    context = 'Deflation is when the general level of prices is falling.'
    question = 'What is deflation?'
    answer = get_answer(context, question)
    assert isinstance(answer, str)
    assert answer == 'when the general level of prices is falling.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_answer()