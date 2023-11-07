from f00232_compute_metrics import *
def test_compute_metrics():
    predictions = np.array([[[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], [[0.3, 0.3, 0.4], [0.2, 0.6, 0.2]]])
    labels = np.array([[0, 1], [2, 1]])
    p = (predictions, labels)
    expected_result = {'precision': 0.5, 'recall': 0.5, 'f1': 0.5, 'accuracy': 0.5}
    assert compute_metrics(p) == expected_result

test_compute_metrics()
