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
    """
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wikisql-supervised')
    answer = model.predict(question, table_data)
    return answer

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
    answer = get_company_revenue(question, table_data)
    assert isinstance(answer, str), 'The function should return a string.'
    assert '$30M' in answer, 'The function should return the correct answer.'

# call_test_function_code --------------------

test_get_company_revenue()