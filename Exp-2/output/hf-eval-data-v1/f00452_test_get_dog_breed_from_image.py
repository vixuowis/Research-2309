def test_get_dog_breed_from_image():
    '''
    This function tests the get_dog_breed_from_image function.
    It uses a sample image URL and a question, and checks if the returned answer is a string.
    '''
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    question = 'What breed are the dogs in the picture?'
    answer = get_dog_breed_from_image(img_url, question)
    assert isinstance(answer, str), 'The function should return a string.'

test_get_dog_breed_from_image()