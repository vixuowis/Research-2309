from typing import *
from torchaudio.transforms import Resample

def preprocess_dataset(dataset):
    resample = Resample(16000)
    dataset = dataset.cast_column("audio", resample)
    return dataset
