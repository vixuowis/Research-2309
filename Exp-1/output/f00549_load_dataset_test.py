from f00549_load_dataset import *
def test_load_cppe5_dataset():
    cppe5 = load_cppe5_dataset()
    assert len(cppe5['train']) == 1000
    assert len(cppe5['test']) == 29
