def test_get_city_probabilities():
    '''
    This function tests the get_city_probabilities function.
    It uses a sample image URL and a list of city names as input.
    It checks if the output is a dictionary and if the sum of the probabilities is approximately 1.
    '''
    # Define a sample image URL and a list of city names
    image_url = 'https://path_to_your_image.com/image.jpg'
    choices = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']

    # Call the get_city_probabilities function
    city_probs = get_city_probabilities(image_url, choices)

    # Check if the output is a dictionary
    assert isinstance(city_probs, dict), 'Output should be a dictionary.'

    # Check if the sum of the probabilities is approximately 1
    assert abs(sum(city_probs.values()) - 1) < 1e-6, 'The sum of the probabilities should be approximately 1.'

test_get_city_probabilities()