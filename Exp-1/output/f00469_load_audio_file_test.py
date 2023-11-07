from f00469_load_audio_file import *
def test_load_audio_file():
    dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
    audio_file, sampling_rate = load_audio_file(dataset, 0)
    assert isinstance(audio_file, str)
    assert isinstance(sampling_rate, int)

test_load_audio_file()
