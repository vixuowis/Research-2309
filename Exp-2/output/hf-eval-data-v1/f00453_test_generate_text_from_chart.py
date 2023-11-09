def test_generate_text_from_chart():
    # Test the function with a sample image
    image_path = 'path_to_test_image.jpg'
    # Replace 'path_to_test_image.jpg' with the path to your test image
    generated_text = generate_text_from_chart(image_path)
    # Assert that the function returns a string
    assert isinstance(generated_text, str), 'The function should return a string.'
    # Assert that the function does not return an empty string
    assert generated_text != '', 'The function should not return an empty string.'

test_generate_text_from_chart()