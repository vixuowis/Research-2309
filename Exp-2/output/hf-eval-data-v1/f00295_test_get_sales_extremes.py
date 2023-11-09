def test_get_sales_extremes():
    """
    This function tests the get_sales_extremes function by using a sample sales data table.
    """
    # Define a sample sales data table
    sales_data_table = [
        {'Date': '2020-01-01', 'Sales': 100},
        {'Date': '2020-01-02', 'Sales': 200},
        {'Date': '2020-01-03', 'Sales': 150},
        {'Date': '2020-01-04', 'Sales': 50}
    ]
    
    # Call the get_sales_extremes function with the sample sales data table
    highest_sales, lowest_sales = get_sales_extremes(sales_data_table)
    
    # Assert that the highest sales number is close to the expected value
    assert abs(highest_sales - 200) < 1e-6, 'Test failed!'
    
    # Assert that the lowest sales number is close to the expected value
    assert abs(lowest_sales - 50) < 1e-6, 'Test failed!'

test_get_sales_extremes()