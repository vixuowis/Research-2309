import os

# Test function for text_to_speech
# @param None
# @return: None

def test_text_to_speech():
    # Test text
    text = 'Hello, world!'
    
    # Expected output file
    expected_output_file = 'output.wav'
    
    # Call the text_to_speech function
    output_file = text_to_speech(text)
    
    # Check that the output file exists
    assert os.path.exists(output_file), f'Output file {output_file} does not exist'
    
    # Check that the output file is the expected one
    assert output_file == expected_output_file, f'Expected output file {expected_output_file}, but got {output_file}'
    
    # TODO: Add more tests, e.g., check the contents of the output file

test_text_to_speech()