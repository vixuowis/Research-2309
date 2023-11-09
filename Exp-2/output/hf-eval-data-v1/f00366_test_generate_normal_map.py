def test_generate_normal_map():
    '''
    This function tests the generate_normal_map function.
    
    Parameters:
    None
    
    Returns:
    None
    '''
    import os
    
    # Define the input and output paths
    input_path = 'path/to/your/test/image.png'
    output_path = 'test_output_normal_map.png'
    
    # Call the function
    generate_normal_map(input_path, output_path)
    
    # Check if the output file is created
    assert os.path.exists(output_path), 'Output file not created.'
    
    # Clean up the output file
    os.remove(output_path)