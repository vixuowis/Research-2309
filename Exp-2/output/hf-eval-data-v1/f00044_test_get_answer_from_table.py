def test_get_answer_from_table():
    '''
    This function tests the get_answer_from_table function.
    '''
    # Define a sample question and table data
    question = 'What is the total revenue for product ID 12345?'
    table_data = {
        'Product ID': ['12345', '67890'],
        'Revenue': [1000, 2000]
    }
    
    # Call the get_answer_from_table function with the sample question and table data
    answer = get_answer_from_table(question, table_data)
    
    # Assert that the function returns the expected answer
    assert answer == '1000', f'Expected 1000, but got {answer}'

test_get_answer_from_table()