from f00541_load_ucf101_dataset import *
def test_load_ucf101_dataset():
    hf_dataset_identifier = "sayakpaul/ucf101-subset"
    filename = "UCF101_subset.tar.gz"
    file_path = load_ucf101_dataset(hf_dataset_identifier, filename)
    assert isinstance(file_path, str)


test_load_ucf101_dataset()
