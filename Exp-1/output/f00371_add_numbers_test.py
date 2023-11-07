from f00371_add_numbers import *
def test_add_numbers():
    # Test case 1
    assert add_numbers(2, 3) == 5

    # Test case 2
    assert add_numbers(-1, 1) == 0

    # Test case 3
    assert add_numbers(0, 0) == 0

    # Test case 4
    assert add_numbers(10, -5) == 5

    # Test case 5
    assert add_numbers(100, 200) == 300


test_add_numbers()
