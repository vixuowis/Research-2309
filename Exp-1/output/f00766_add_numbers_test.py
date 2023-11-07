from f00766_add_numbers import *
import numpy as np
import pytest


# Test cases
def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-5, 10) == 5
    assert add_numbers(0, 0) == 0
    assert add_numbers(100, -50) == 50
    assert add_numbers(7, -7) == 0


def test_entry():
    pytest.main([__file__])
