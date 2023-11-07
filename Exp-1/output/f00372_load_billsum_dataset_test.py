from f00372_load_billsum_dataset import *
def test_load_billsum_dataset():
    billsum = load_billsum_dataset()
    assert isinstance(billsum, dict)
    assert "train" in billsum
    assert "validation" in billsum
    assert "test" in billsum
    assert len(billsum["train"]) > 0
    assert len(billsum["validation"]) > 0
    assert len(billsum["test"]) > 0


def test_entry():
    test_load_billsum_dataset()
