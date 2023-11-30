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
    
    # Create pipeline using deepset/roberta-base-squad2 model
    nlp = pipeline("question-answering", model="deepset/roberta-large-squad2")

    context = "The capital of Germany is Berlin."
    question = "What is the capital of Germany?"
    
    answer = nlp({'question': question, 'context': context})
    return(answer)

# test_function_code --------------------

def test_get_capital_of_germany():
    """
    This function tests the 'get_capital_of_germany' function by comparing the output to the expected answer 'Berlin'.
    """
    assert get_capital_of_germany() == 'Berlin'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_get_capital_of_germany()