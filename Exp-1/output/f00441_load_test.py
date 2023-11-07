from f00441_load import *
import evaluate
import assert

def test_load():
    metric_name = "accuracy"
    expected_metric = evaluate.load(metric_name)
    assert load(metric_name) == expected_metric

    metric_name = "precision"
    expected_metric = evaluate.load(metric_name)
    assert load(metric_name) == expected_metric

    metric_name = "recall"
    expected_metric = evaluate.load(metric_name)
    assert load(metric_name) == expected_metric
