from f00490_compute_metrics import *
import numpy as np


def test_compute_metrics():
    eval_pred = (np.array([[0.1, 0.9], [0.3, 0.7], [0.6, 0.4]]), np.array([1, 0, 1]))
    assert compute_metrics(eval_pred) == 0.6666666666666666

test_compute_metrics()
