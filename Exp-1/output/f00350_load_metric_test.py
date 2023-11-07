from f00350_load_metric import *
import evaluate
import assert

metric = evaluate.load('sacrebleu')

assert isinstance(metric, evaluate.Metric)
assert metric.name == 'sacrebleu'
