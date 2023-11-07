from f00199_load import *
import evaluate
import assert


# Test case 1
metric_name = "accuracy"
expected_output = evaluate.load(metric_name)
assert expected_output == accuracy

# Test case 2
metric_name = "precision"
expected_output = evaluate.load(metric_name)
assert expected_output == precision

# Test case 3
metric_name = "recall"
expected_output = evaluate.load(metric_name)
assert expected_output == recall

# Test case 4
metric_name = "f1_score"
expected_output = evaluate.load(metric_name)
assert expected_output == f1_score

# Test case 5
metric_name = "roc_auc_score"
expected_output = evaluate.load(metric_name)
assert expected_output == roc_auc_score
