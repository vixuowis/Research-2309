from f00451_add_numbers import *
def test_add_numbers():
    assert add_numbers([1, 2, 3]) == 6
    assert add_numbers([4, 5, 6]) == 15
    assert add_numbers([-1, 0, 1]) == 0
    assert add_numbers([]) == 0
    assert add_numbers([10]) == 10

test_add_numbers()
