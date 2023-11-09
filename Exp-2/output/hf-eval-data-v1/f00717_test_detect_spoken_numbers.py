# Test function for detect_spoken_numbers
# This function uses a sample audio file to test the detect_spoken_numbers function
# The test asserts that the function returns a list

def test_detect_spoken_numbers():
    # Path to a sample audio file
    sample_audio_file_path = 'sample_audio_file.pt'
    
    # Call the function with the sample audio file
    detected_digits = detect_spoken_numbers(sample_audio_file_path)
    
    # Assert that the function returns a list
    assert isinstance(detected_digits, list), 'The function should return a list.'
    
    # Assert that the list is not empty
    assert len(detected_digits) > 0, 'The function should detect at least one digit.'
    
    # Assert that the detected digits are in the range 0-9
    assert all(0 <= digit <= 9 for digit in detected_digits), 'All detected digits should be in the range 0-9.'
    
    print('All tests passed.')

# Run the test function
test_detect_spoken_numbers()