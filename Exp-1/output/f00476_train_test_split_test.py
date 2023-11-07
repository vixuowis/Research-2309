from f00476_train_test_split import *
def test_train_test_split():
    dataset = Dataset.from_dict({
        'train': [{'text': 'This is a sample sentence.'}, {'text': 'Another sample sentence.'}],
        'test': [{'text': 'Test sentence 1.'}, {'text': 'Test sentence 2.'}]}
    )
    train, test = dataset.train_test_split(test_size=0.2)

    assert len(train) == 2
    assert len(test) == 1
    assert train[0]['text'] == 'This is a sample sentence.'
    assert test[0]['text'] == 'Test sentence 1.'

test_train_test_split()
