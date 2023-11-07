from f00508_add_numbers import *
def test_add_numbers():
    # Test case 1
    numbers = [1, 2, 3, 4, 5]
    expected_output = 15
    assert add_numbers(numbers) == expected_output
    
    # Test case 2
    numbers = [10, 20, 30]
    expected_output = 60
    assert add_numbers(numbers) == expected_output
    
    # Test case 3
    numbers = []
    expected_output = 0
    assert add_numbers(numbers) == expected_output
