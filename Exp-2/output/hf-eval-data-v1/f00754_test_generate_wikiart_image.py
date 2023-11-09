def test_generate_wikiart_image():
    """
    This function tests the generate_wikiart_image function by generating an image and checking if the output is not None.
    """
    image = generate_wikiart_image()
    assert image is not None, 'The generated image should not be None.'