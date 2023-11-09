def test_get_tallest_animal():
    """
    This function tests the 'get_tallest_animal' function.
    It uses a sample dataset of animals and their characteristics.
    It uses the 'assert' statement to ensure the function's output is as expected.
    """
    # Sample dataset
    animal_table = [['Animal', 'Height'], ['Giraffe', '5.5m'], ['Elephant', '3.3m'], ['Lion', '1.2m']]
    # Expected output
    expected_output = 'Giraffe'
    # Get function output
    function_output = get_tallest_animal(animal_table)
    # Assert function output is as expected
    assert function_output == expected_output, f'Expected {expected_output}, but got {function_output}'

test_get_tallest_animal()