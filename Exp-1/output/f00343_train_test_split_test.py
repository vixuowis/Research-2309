from f00343_train_test_split import *
def test_train_test_split():
    dataset = Dataset.from_dict({"input": [1, 2, 3, 4, 5], "output": [6, 7, 8, 9, 10]})
    train_set, test_set = train_test_split(dataset, test_size=0.2)
    assert len(train_set) == 4
    assert len(test_set) == 1
    assert train_set["input"] == [1, 2, 3, 4]
    assert test_set["input"] == [5]
    assert train_set["output"] == [6, 7, 8, 9]
    assert test_set["output"] == [10]

test_train_test_split()
