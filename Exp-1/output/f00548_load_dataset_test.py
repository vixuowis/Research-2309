from f00548_load_dataset import *
def test_load_dataset():
    dataset = load_dataset('imdb')
    assert len(dataset) > 0
    assert 'text' in dataset[0]
    assert 'label' in dataset[0]


test_load_dataset()
