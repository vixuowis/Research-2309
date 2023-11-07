from f00595_add_numbers import *
def test_add_numbers():
    assert add_numbers([1, 2, 3]) == 6, 'Test Case 1 Failed'
    assert add_numbers([-1, 0, 1]) == 0, 'Test Case 2 Failed'
    assert add_numbers([]) == 0, 'Test Case 3 Failed'
    assert add_numbers([10, 20, 30, 40, 50]) == 150, 'Test Case 4 Failed'
    assert add_numbers([-5, -10, -15, -20]) == -50, 'Test Case 5 Failed'

test_add_numbers()
