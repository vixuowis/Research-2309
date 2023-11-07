from f00115_load import *
import numpy as np
import evaluate


def test_load():
    metric = evaluate.load("accuracy")
    assert isinstance(metric, Callable)
    assert isinstance(metric(np.array([1, 2, 3]), np.array([1, 2, 3])), float)

test_load()
