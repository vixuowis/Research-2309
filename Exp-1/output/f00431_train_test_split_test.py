from f00431_train_test_split import *
def test_train_test_split():
    dataset = Dataset.from_dict({'train': {'text': ['example 1', 'example 2', 'example 3']}})
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
    assert len(train_dataset) == 2
    assert len(test_dataset) == 1
    assert train_dataset[0]['text'] == 'example 1'
    assert train_dataset[1]['text'] == 'example 2'
    assert test_dataset[0]['text'] == 'example 3'


test_train_test_split()
