from f00283_train_test_split import *
def test_train_test_split():
    dataset = Dataset()
    train_set, test_set = train_test_split(dataset, test_size=0.2)
    assert len(train_set) == int(len(dataset) * 0.8)
    assert len(test_set) == int(len(dataset) * 0.2)


test_train_test_split()
