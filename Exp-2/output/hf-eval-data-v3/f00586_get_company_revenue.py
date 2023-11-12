# function_import --------------------

from transformers import TapasForQuestionAnswering

# function_code --------------------

def get_company_revenue(question: str, table_data: list) -> str:
    """
    This function uses the TAPAS model to answer questions based on tabular data.

    Args:
        question (str): The question to be answered.
        table_data (list): The table data in the form of a list of dictionaries.

    Returns:
        str: The answer to the question.

    Raises:
        OSError: If there is an issue with loading the model or the disk quota is exceeded.
    """
    try:
        model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wikisql-supervised')
        answer = model.predict(question, table_data)
        return answer
    except OSError as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_get_company_revenue():
    """
    This function tests the get_company_revenue function.
    """
    question = 'What was the revenue of the company in 2020?'
    table_data = [
        {'Year': '2018', 'Revenue': '$20M'},
        {'Year': '2019', 'Revenue': '$25M'},
        {'Year': '2020', 'Revenue': '$30M'},
    ]
    assert isinstance(get_company_revenue(question, table_data), str)
    
    question = 'What was the revenue of the company in 2019?'
    assert isinstance(get_company_revenue(question, table_data), str)
    
    question = 'What was the revenue of the company in 2018?'
    assert isinstance(get_company_revenue(question, table_data), str)
    
    print('All Tests Passed')

# call_test_function_code --------------------

test_get_company_revenue()