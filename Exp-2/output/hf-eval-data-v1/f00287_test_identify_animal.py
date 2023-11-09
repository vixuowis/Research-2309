def test_identify_animal():
    """
    This function tests the identify_animal function.
    It uses an online image of a cat and a dog for testing.
    """
    # Define the image paths
    cat_image_path = 'https://example.com/cat.jpg'
    dog_image_path = 'https://example.com/dog.jpg'
    
    # Test the function
    assert identify_animal(cat_image_path) == 'cat'
    assert identify_animal(dog_image_path) == 'dog'