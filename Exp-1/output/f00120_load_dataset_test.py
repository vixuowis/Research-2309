from f00120_load_dataset import *
def test_load_dataset():
    dataset = load_dataset("glue", "cola")
    assert isinstance(dataset, Dataset)

test_load_dataset()
