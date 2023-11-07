from typing import *
import evaluate

def load_evaluation_metric(metric_name: str) -> object:
    return evaluate.load(metric_name)
