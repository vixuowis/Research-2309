from f00221_add_numbers import *
def test_add_numbers():
    assert add_numbers([1, 2, 3]) == 6
    assert add_numbers([10, 20, 30]) == 60
    assert add_numbers([-1, -2, -3]) == -6
    assert add_numbers([]) == 0
    assert add_numbers([5]) == 5

test_add_numbers()
