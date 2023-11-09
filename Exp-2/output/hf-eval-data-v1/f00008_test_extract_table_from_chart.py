def test_extract_table_from_chart():
    '''
    This function tests the extract_table_from_chart function.
    It uses a sample chart image URL and checks if the output is a string.
    '''
    # Sample chart image URL
    chart_url = 'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png'
    
    # Call the function with the sample chart image URL
    result = extract_table_from_chart(chart_url)
    
    # Check if the output is a string
    assert isinstance(result, str), 'The output should be a string.'
    
    # Print the result for manual inspection
    print(result)

test_extract_table_from_chart()