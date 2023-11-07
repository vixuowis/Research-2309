from f00429_add_numbers import *
import numpy as np
import pytest


def test_add_numbers():
    # Test case 1
    result = add_numbers(2, 3)
    assert result == 5

    # Test case 2
    result = add_numbers(0, 0)
    assert result == 0

    # Test case 3
    result = add_numbers(-5, 5)
    assert result == 0

    # Test case 4
    result = add_numbers(10, -2)
    assert result == 8

    # Test case 5
    result = add_numbers(100, 200)
    assert result == 300

