def test_get_total_sales():
    '''
    This function tests the get_total_sales function.
    It uses a sample sales data table and a question, and checks if the returned total sales is within a reasonable range.
    '''
    # Sample sales data table
    sales_data_table = [['Product', 'Week 1', 'Week 2', 'Week 3'], ['Product A', 100, 200, 300], ['Product B', 150, 250, 350]]
    
    # Sample question
    question = 'What is the total sales of Product A?'
    
    # Call the function with the sample data
    total_sales = get_total_sales(sales_data_table, question)
    
    # Check if the returned total sales is within a reasonable range
    assert 500 <= total_sales <= 700, 'The total sales is not within the expected range'