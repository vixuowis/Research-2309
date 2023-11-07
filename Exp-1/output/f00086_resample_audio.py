from typing import *
from datasets import Dataset, Audio

def resample_audio(dataset: Dataset) -> Dataset:
    # Resample audio column to 16kHz
    # Args:
    #     dataset (Dataset): The dataset to resample
    # Returns:
    #     Dataset: The resampled dataset
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    return dataset
