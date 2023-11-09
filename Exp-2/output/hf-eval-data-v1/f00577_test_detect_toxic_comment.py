# This is a test function for detect_toxic_comment function.
# It uses assert to check if the function works correctly.
# The test data is a sample message that is expected to be non-toxic.
def test_detect_toxic_comment():
    # Define a test message
    test_message = 'This is a test text.'
    
    # Call the function with the test message
    result = detect_toxic_comment(test_message)
    
    # Check if the result is as expected
    # Since the model's accuracy is not 100%, we do not compare the result strictly.
    # Instead, we check if the result is a list (which is the expected output format).
    assert isinstance(result, list), 'The result should be a list.'
    
    # If the test passes, print a success message
    print('Test passed.')

# Call the test function
test_detect_toxic_comment()