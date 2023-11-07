from f00464_load import *
import evaluate
import assert

def test_load():
    metric_name = "wer"
    expected_output = evaluate.load(metric_name)
    assert load(metric_name) == expected_output

test_load()
