def test_get_image_caption():
    """
    This function tests the get_image_caption function.
    It uses a sample image and question, and checks if the returned answer is of type str.
    """
    image_path = 'path_to_test_image.jpg'
    question = 'Jakie są główne kolory na zdjęciu?'
    answer = get_image_caption(image_path, question)
    assert isinstance(answer, str), 'The function should return a string.'

test_get_image_caption()