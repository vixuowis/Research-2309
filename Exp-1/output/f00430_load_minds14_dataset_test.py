from f00430_load_minds14_dataset import *
def test_load_minds14_dataset():
    dataset = load_minds14_dataset()
    assert len(dataset) > 0

    sample = dataset[0]
    assert "audio" in sample
    assert "transcript" in sample

    print("All tests passed!")
