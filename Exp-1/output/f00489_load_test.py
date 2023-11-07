from f00489_load import *
import evaluate
import assert


# Test 1
metric_name = 'accuracy'
expected = evaluate.load(metric_name)
assert load(metric_name) == expected

# Test 2
metric_name = 'precision'
expected = evaluate.load(metric_name)
assert load(metric_name) == expected

# Test 3
metric_name = 'recall'
expected = evaluate.load(metric_name)
assert load(metric_name) == expected

