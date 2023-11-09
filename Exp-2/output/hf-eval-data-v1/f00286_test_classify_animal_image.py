def test_classify_animal_image():
    """
    This function tests the 'classify_animal_image' function with a few example images.
    """
    # Define the test images and their expected categories
    test_images = [
        ('https://example.com/cat.jpg', 'cat'),
        ('https://example.com/dog.jpg', 'dog'),
        ('https://example.com/bird.jpg', 'bird')
    ]
    
    # Test the function with each image
    for image_url, expected_category in test_images:
        predicted_category = classify_animal_image(image_url)
        assert predicted_category == expected_category, f'Expected {expected_category}, but got {predicted_category}'

test_classify_animal_image()