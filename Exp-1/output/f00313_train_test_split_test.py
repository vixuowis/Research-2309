from f00313_train_test_split import *
def test_train_test_split():
    dataset = Dataset()
    # Add data to the dataset
    test_size = 0.2
    train_set, test_set = train_test_split(dataset, test_size)
    assert len(train_set) == int(len(dataset) * (1 - test_size))
    assert len(test_set) == int(len(dataset) * test_size)


test_train_test_split()
