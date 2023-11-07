from f00373_train_test_split import *
def test_train_test_split():
    train_dataset, test_dataset = train_test_split(0.2)
    assert len(train_dataset) > 0
    assert len(test_dataset) > 0
    assert len(train_dataset) + len(test_dataset) == len(billsum)


test_train_test_split()
