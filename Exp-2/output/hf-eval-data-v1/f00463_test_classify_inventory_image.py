def test_classify_inventory_image():
    """
    This function tests the classify_inventory_image function by using a test image.
    """
    test_image_path = 'test_image.jpg'  # replace with the path to your test image
    predicted_label = classify_inventory_image(test_image_path)
    print(f'The predicted label for the test image is: {predicted_label}')
    assert isinstance(predicted_label, str), 'The predicted label should be a string.'

test_classify_inventory_image()