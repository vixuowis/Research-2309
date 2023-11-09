def test_get_mars_orbit_duration():
    '''
    This function tests the 'get_mars_orbit_duration' function.
    '''
    # Define the context and the question
    context = 'Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, being larger than only Mercury. Mars takes approximately 687 Earth days to complete one orbit around the Sun.'
    question = 'How long does it take for Mars to orbit the sun?'
    
    # Call the 'get_mars_orbit_duration' function
    answer = get_mars_orbit_duration(context, question)
    
    # Assert that the answer is approximately correct
    assert '687' in answer, f'Error: {answer}'
    
    # Print a success message
    print('All tests passed.')

test_get_mars_orbit_duration()