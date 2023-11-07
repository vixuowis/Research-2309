from f00510_train_test_split import *
def test_train_test_split():
    dataset = Dataset()
    test_size = 0.2
    train_ds, test_ds = train_test_split(dataset, test_size)

    assert len(train_ds) == int(len(dataset) * (1 - test_size))
    assert len(test_ds) == int(len(dataset) * test_size)

    # Add more test cases if needed


test_train_test_split()
