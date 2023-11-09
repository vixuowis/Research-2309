def test_generate_textual_description():
    # Test the generate_textual_description function
    # Use a sample image for testing
    image_path = 'sample_image.jpg'
    # Generate textual description for the sample image
    description = generate_textual_description(image_path)
    # Check if the generated description is a string
    assert isinstance(description, str), 'The generated description should be a string.'
    # Check if the generated description is not empty
    assert len(description) > 0, 'The generated description should not be empty.'

test_generate_textual_description()