from f00119_multiply import *
def test_multiply():
    assert multiply([1, 2, 3, 4]) == 24
    assert multiply([-1, 2, 3, -4]) == -24
    assert multiply([0, 1, 2, 3]) == 0
    assert multiply([]) == 1
    assert multiply([5]) == 5

test_multiply()
