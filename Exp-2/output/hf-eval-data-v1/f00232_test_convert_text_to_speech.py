def test_convert_text_to_speech():
    '''
    This function tests the convert_text_to_speech function by using a sample Chinese text.
    
    Parameters:
    None
    
    Returns:
    None
    '''
    # Define a sample Chinese text
    text = '春江潮水连海平，海上明月共潮生'
    
    # Call the function with the sample text
    convert_text_to_speech(text)
    
    # Check if the output file has been created
    assert os.path.exists('out.wav')
    
    # If the file exists, delete it for cleanup
    if os.path.exists('out.wav'):
        os.remove('out.wav')

test_convert_text_to_speech()