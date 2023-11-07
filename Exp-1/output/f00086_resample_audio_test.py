from f00086_resample_audio import *
def test_resample_audio():
    dataset = Dataset.from_dict({"audio": [array([1, 2, 3]), array([4, 5, 6])], "label": [0, 1]})
    resampled_dataset = resample_audio(dataset)
    assert resampled_dataset["audio"][0].shape[0] == 48000
    assert resampled_dataset["audio"][1].shape[0] == 48000
    assert resampled_dataset["label"] == dataset["label"]

test_resample_audio()
