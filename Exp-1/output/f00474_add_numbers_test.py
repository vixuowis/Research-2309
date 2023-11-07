from f00474_add_numbers import *
def test_add_numbers():
    assert add_numbers([1, 2, 3, 4, 5]) == 15
    assert add_numbers([10, 20, 30, 40, 50]) == 150
    assert add_numbers([-1, -2, -3, -4, -5]) == -15
    assert add_numbers([0, 0, 0, 0, 0]) == 0
    assert add_numbers([]) == 0

test_add_numbers()
