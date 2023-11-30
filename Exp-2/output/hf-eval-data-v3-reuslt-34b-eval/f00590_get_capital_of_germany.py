# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_capital_of_germany():
    """
    This function uses the Hugging Face Transformers library to answer the question "What is the capital of Germany?".
    It uses the 'deepset/roberta-large-squad2' model for question answering.
    
    Returns:
        str: The capital of Germany.
    """
    qa_pipeline = pipeline('question-answering', model='deepset/roberta-large-squad2', tokenizer='deepset/roberta-large-squad2')
    
    context = "German states may call extraordinay state legislature"
    question = "What is the capital of Germany?"

    result = qa_pipeline(question=question, context=context)
    
    return result['answer']


# test_function_code --------------------

def test_get_capital_of_germany():
    """
    This function tests the 'get_capital_of_germany' function by comparing the output to the expected answer 'Berlin'.
    """
    assert get_capital_of_germany() == 'Berlin'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_get_capital_of_germany()