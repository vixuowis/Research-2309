def test_generate_response():
    '''
    This function tests the generate_response function.
    It uses a sample dataset from Kaggle game script dataset.
    '''
    # Load the dataset
    dataset = pd.read_csv('Kaggle game script dataset.csv')
    # Select a sample from the dataset
    sample = dataset.sample(n=1)
    # Get the user input from the sample
    user_input = sample['user_input'].values[0]
    # Generate a response
    response = generate_response(user_input)
    # Check if the response is a string
    assert isinstance(response, str), 'The response should be a string.'
    # Check if the response is not empty
    assert response != '', 'The response should not be empty.'

test_generate_response()