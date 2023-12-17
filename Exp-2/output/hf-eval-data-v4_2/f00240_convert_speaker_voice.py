# requirements_file --------------------

!pip install -U transformers datasets numpy torch soundfile

# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import numpy as np
import torch
import soundfile as sf

# function_code --------------------

def convert_speaker_voice(audio_data, sr, speaker_embedding_path):
    """
    Converts a speaker's voice in given audio data to another, using pretrained SpeechT5 models.

    Args:
        audio_data (np.ndarray): The audio data array.
        sr (int): The sampling rate of the audio data.
        speaker_embedding_path (str): The path to the numpy file containing the speaker embeddings.

    Returns:
        np.ndarray: The audio data converted to the new speaker's voice.

    Raises:
        FileNotFoundError: If the provided speaker embedding file does not exist.
    """
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_vc')
    model = SpeechT5ForSpeechToSpeech.from_pretrained('microsoft/speecht5_vc')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

    inputs = processor(audio=audio_data, sampling_rate=sr, return_tensors='pt')
    speaker_embeddings = np.load(speaker_embedding_path)
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
    speech = model.generate_speech(inputs['input_values'], speaker_embeddings, vocoder=vocoder)

    sf.write('speech.wav', speech.numpy(), samplerate=sr)
    return speech.numpy()

# test_function_code --------------------

def test_convert_speaker_voice():
    print("Testing started.")
    dataset = load_dataset('hf-internal-testing/librispeech_asr_demo', 'clean', split='validation')
    sample_audio = dataset[0]['audio']['array']
    sample_rate = dataset.features['audio'].sampling_rate

    print("Testing case [1/1] started.")
    try:
        converted_audio = convert_speaker_voice(sample_audio, sample_rate, 'xvector_speaker_embedding.npy')
        assert isinstance(converted_audio, np.ndarray), "The returned object is not a numpy array."
        assert converted_audio.shape == sample_audio.shape, "The shape of the converted audio does not match the original audio."
    except FileNotFoundError as e:
        print(f"Test case [1/1] failed: {e}")
    print("Testing finished.")


# call_test_function_line --------------------

test_convert_speaker_voice()