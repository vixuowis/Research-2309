from f00311_calculate_average import *
def test_calculate_average():
    assert calculate_average([1, 2, 3, 4, 5]) == 3.0
    assert calculate_average([10, 20, 30, 40, 50]) == 30.0
    assert calculate_average([-1, 0, 1]) == 0.0
    assert calculate_average([]) == 0.0
    assert calculate_average([5]) == 5.0

test_calculate_average()
