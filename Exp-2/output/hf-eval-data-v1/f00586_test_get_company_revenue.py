def test_get_company_revenue():
    """
    This function tests the get_company_revenue function.
    It uses a sample dataset and a sample question.
    """
    # Define the sample question and table data
    question = 'What was the revenue of the company in 2020?'
    table_data = [
      {'Year': '2018', 'Revenue': '$20M'},
      {'Year': '2019', 'Revenue': '$25M'},
      {'Year': '2020', 'Revenue': '$30M'},
    ]
    
    # Call the function with the sample data
    answer = get_company_revenue(question, table_data)
    
    # Assert that the function returns the correct answer
    assert answer == '$30M', f'Error: {answer}'

test_get_company_revenue()