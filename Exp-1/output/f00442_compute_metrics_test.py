from f00442_compute_metrics import *
def test_compute_metrics():
    eval_pred = {
        'predictions': np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.1, 0.4]]),
        'label_ids': np.array([2, 1, 0])
    }
    assert compute_metrics(eval_pred) == 1.0

test_compute_metrics()
