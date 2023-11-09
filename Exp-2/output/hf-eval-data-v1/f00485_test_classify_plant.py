def test_classify_plant():
    """
    Function to test the classify_plant function.
    """
    # Define the image path and labels
    image_path = 'path/to/plant_image.jpg'
    labels = ['rose', 'tulip', 'sunflower']
    
    # Call the classify_plant function
    result = classify_plant(image_path, labels)
    
    # Assert that the result is in the labels
    assert result in labels, f'Error: {result} not in {labels}'
    
    print('All tests passed.')

test_classify_plant()