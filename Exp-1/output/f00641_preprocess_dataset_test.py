from f00641_preprocess_dataset import *
def test_preprocess_dataset():
    dataset = DummyDataset()
    preprocessed_dataset = preprocess_dataset(dataset)
    assert preprocessed_dataset == expected_dataset


def DummyDataset():
    return {...}


def expected_dataset():
    return {...}


test_preprocess_dataset()
