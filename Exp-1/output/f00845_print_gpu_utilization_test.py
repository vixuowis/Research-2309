from f00845_print_gpu_utilization import *
import torch
import pytest


@pytest.fixture

def test_print_gpu_utilization():
    print_gpu_utilization()
    assert True
