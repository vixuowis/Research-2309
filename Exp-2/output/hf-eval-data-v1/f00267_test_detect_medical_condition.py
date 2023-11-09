# Test function for detect_medical_condition
# Uses a test image and checks if the output is a string (as the medical condition is expected to be a string)
# Does not compare the output strictly as the model's output can vary based on the input image
def test_detect_medical_condition():
    test_image = 'test_image.jpg' # replace with a valid test image
    output = detect_medical_condition(test_image)
    assert isinstance(output, str), 'Output should be a string'
    print('Test passed.')

# Run the test function
test_detect_medical_condition()