from f00381_compute_metrics import *
import numpy as np


def test_compute_metrics():
    eval_pred = (predictions, labels)
    result = compute_metrics(eval_pred)
    assert result == expected_result


def main():
    test_compute_metrics()


if __name__ == '__main__':
    main()
