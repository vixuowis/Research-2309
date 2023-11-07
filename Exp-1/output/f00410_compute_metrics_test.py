from f00410_compute_metrics import *
import numpy as np


def test_compute_metrics():
    eval_pred = (np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]]), np.array([2, 0, 0]))
    expected_accuracy = 0.6666666666666666
    assert compute_metrics(eval_pred) == expected_accuracy

test_compute_metrics()
