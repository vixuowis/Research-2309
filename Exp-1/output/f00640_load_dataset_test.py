from f00640_load_dataset import *
def test_load_voxpopuli_dataset():
    dataset = load_voxpopuli_dataset('nl')
    assert len(dataset) == 20968

def test_load_voxpopuli_dataset_invalid_language():
    dataset = load_voxpopuli_dataset('invalid_language')
    assert dataset is None


def test_load_voxpopuli_dataset_empty_language():
    dataset = load_voxpopuli_dataset('')
    assert dataset is None


def test_load_voxpopuli_dataset_none_language():
    dataset = load_voxpopuli_dataset(None)
    assert dataset is None
