def test_extract_data_from_plot():
    """
    This function tests the extract_data_from_plot function.
    It uses a sample image of a plot and checks if the extracted data table is not empty.
    """
    # Define the path to the sample image
    sample_image_path = 'sample_plot.png'
    
    # Call the function with the sample image
    result = extract_data_from_plot(sample_image_path)
    
    # Check if the result is not empty
    assert result, 'The extracted data table is empty.'
    
    print('The test passed successfully.')

test_extract_data_from_plot()