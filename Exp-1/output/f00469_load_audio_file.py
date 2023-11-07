from typing import *
from datasets import load_dataset, Audio

def load_audio_file(dataset, index):
    """This function takes a dataset and an index, and returns the path of the audio file at the given index in the dataset, as well as the sampling rate of the audio file.

    Args:
        dataset (Dataset): The dataset containing the audio files.
        index (int): The index of the audio file in the dataset.

    Returns:
        audio_file (str): The path of the audio file.
        sampling_rate (int): The sampling rate of the audio file."""
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    sampling_rate = dataset.features["audio"].sampling_rate
    audio_file = dataset[index]["audio"]["path"]
    return audio_file, sampling_rate
