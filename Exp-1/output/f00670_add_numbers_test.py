from f00670_add_numbers import *
import numpy as np
import pytest


def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0
    assert add_numbers(10, -5) == 5
    assert add_numbers(100, 200) == 300


if __name__ == '__main__':
    pytest.main([__file__])
