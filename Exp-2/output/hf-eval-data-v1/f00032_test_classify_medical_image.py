def test_classify_medical_image():
    """
    This function tests the 'classify_medical_image' function by using a sample medical image.
    """
    # Specify the path to the sample medical image
    image_path = 'path/to/sample_medical_image.png'
    
    # Call the 'classify_medical_image' function
    result = classify_medical_image(image_path)
    
    # Assert that the result is one of the possible class names
    assert result in ['X-ray', 'MRI scan', 'CT scan'], f'Unexpected result: {result}'

test_classify_medical_image()