from typing import *
import torchaudio


def prepare_dataset(example):
    # Load audio waveform
    waveform, sample_rate = torchaudio.load(example['file'])
    
    # Convert waveform to mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)
    
    return {
        'file': example['file'],
        'spectrogram': mel_spectrogram
    }
