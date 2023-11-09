def test_get_animal_info():
    '''
    This function tests the 'get_animal_info' function.
    It uses a set of predefined questions and compares the output of the function to the expected answers.
    '''
    # Define the test queries and their expected answers
    test_queries = [
        'What is the average lifespan of a giraffe?',
        'What is the habitat of a lion?',
        'What is the average lifespan of a tiger?',
    ]
    expected_answers = [
        '25',
        'Grassland',
        '10',
    ]

    # Test the function
    for i, query in enumerate(test_queries):
        assert str(get_animal_info(query)[0]) in expected_answers[i]

# Run the test function
test_get_animal_info()