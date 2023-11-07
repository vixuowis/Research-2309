from f00409_load import *
import evaluate
import assert

def test_load():
    assert accuracy == evaluate.load("accuracy")
    assert isinstance(accuracy, Any)
    assert accuracy.__name__ == "accuracy"
    assert accuracy.__module__ == "evaluate"
    assert accuracy.__class__.__name__ == "function"


test_load()
