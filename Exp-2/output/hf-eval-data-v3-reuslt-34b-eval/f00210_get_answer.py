# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(question: str, context: str) -> str:
    """
    This function uses the Hugging Face Transformers library to answer a question based on a given context.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the answer is to be found.

    Returns:
        str: The answer to the question based on the context.
    """
    
    nlp = pipeline('question-answering')
    output = nlp({'context': context, 'question': question})
    return output['answer']

# test_function_code --------------------

def test_get_answer():
    assert get_answer('What is the capital of Sweden?', 'Stockholm is the beautiful capital of Sweden, which is known for its high living standards and great attractions.') == 'Stockholm'
    assert get_answer('Who won the world cup in 2018?', 'The 2018 FIFA World Cup was won by France.') == 'France'
    assert get_answer('Who is the president of the United States?', 'As of 2021, the president of the United States is Joe Biden.') == 'Joe Biden'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_get_answer()