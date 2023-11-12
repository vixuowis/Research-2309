# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
import numpy as np
import soundfile as sf
import torch
import os

# function_code --------------------

def convert_voice(input_audio_path: str, speaker_embedding_path: str, output_audio_path: str):
    """
    Convert a recorded customer call into an audio file with a different voice without changing the content.

    Args:
        input_audio_path (str): Path to the input audio file.
        speaker_embedding_path (str): Path to the speaker embedding file.
        output_audio_path (str): Path to save the output audio file.

    Returns:
        None
    """
    example_speech, sampling_rate = sf.read(input_audio_path)
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_vc')
    model = SpeechT5ForSpeechToSpeech.from_pretrained('microsoft/speecht5_vc')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

    inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors='pt')
    speaker_embeddings = np.load(speaker_embedding_path)
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
    speech = model.generate_speech(inputs['input_values'], speaker_embeddings, vocoder=vocoder)

    sf.write(output_audio_path, speech.numpy(), samplerate=16000)

# test_function_code --------------------

def test_convert_voice():
    """
    Test the convert_voice function.
    """
    # Prepare test data
    input_audio_path = 'test_input_audio.wav'
    speaker_embedding_path = 'test_xvector_speaker_embedding.npy'
    output_audio_path = 'test_converted_speech.wav'

    # Call the function
    convert_voice(input_audio_path, speaker_embedding_path, output_audio_path)

    # Check the output
    assert os.path.exists(output_audio_path), 'Output audio file not found.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_convert_voice()