from f00380_load_evaluation_metric import *
def test_load_evaluation_metric():
    rouge = load_evaluation_metric('rouge')
    assert isinstance(rouge, object)

test_load_evaluation_metric()
