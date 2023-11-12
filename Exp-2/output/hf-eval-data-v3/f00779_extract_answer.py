# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_answer(question: str, context: str) -> str:
    """
    Extracts the answer to a given question from a given context using a pre-trained model.

    Args:
        question (str): The question to answer.
        context (str): The context from which to extract the answer.

    Returns:
        str: The extracted answer.
    """
    nlp = pipeline('question-answering', model='deepset/deberta-v3-large-squad2', tokenizer='deepset/deberta-v3-large-squad2')
    QA_input = {'question': question, 'context': context}
    result = nlp(QA_input)
    return result['answer']

# test_function_code --------------------

def test_extract_answer():
    """
    Tests the extract_answer function.
    """
    question = 'What is the penalty for breaking the contract?'
    context = 'The penalty for breaking the contract is generally...'
    assert isinstance(extract_answer(question, context), str)
    question = 'Who is the CEO of the company?'
    context = 'The CEO of the company is John Doe.'
    assert extract_answer(question, context) == 'John Doe'
    question = 'When was the company founded?'
    context = 'The company was founded in 2000.'
    assert extract_answer(question, context) == '2000'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_answer()