from f00116_compute_metrics import *
def test_compute_metrics():
    eval_pred = (logits, labels)
    expected_result = metric.compute(predictions=np.argmax(logits, axis=-1), references=labels)
    assert compute_metrics(eval_pred) == expected_result


test_compute_metrics()
