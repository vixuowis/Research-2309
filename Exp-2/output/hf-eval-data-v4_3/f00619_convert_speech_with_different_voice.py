# requirements_file --------------------

import subprocess

requirements = ["transformers", "datasets", "numpy", "torch", "soundfile"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
import numpy as np
import soundfile as sf
import torch

# function_code --------------------

def convert_speech_with_different_voice(input_audio_path, speaker_embedding_path, output_audio_path):
    """Convert a recorded speech to a different voice without changing the content.

    Args:
        input_audio_path (str): The file path to the input audio wav file.
        speaker_embedding_path (str): The file path to the numpy file containing x-vector speaker embedding.
        output_audio_path (str): The file path where the converted speech will be saved.

    Returns:
        str: The file path to the converted audio.

    Raises:
        FileNotFoundError: If either input_audio_path or speaker_embedding_path does not exist.
        Exception: If conversion fails.
    """
    # Load the example speech from the input file and retrieve the sampling rate
    try:
        example_speech, sampling_rate = sf.read(input_audio_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'Input audio file not found: {input_audio_path}')

    # Load the speaker embeddings
    try:
        speaker_embeddings = np.load(speaker_embedding_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'Speaker embedding file not found: {speaker_embedding_path}')
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

    # Create instances of SpeechT5Processing classes
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_vc')
    model = SpeechT5ForSpeechToSpeech.from_pretrained('microsoft/speecht5_vc')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

    # Preprocess the input audio
    inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors='pt')

    # Generate the converted speech
    try:
        speech = model.generate_speech(inputs['input_values'], speaker_embeddings, vocoder=vocoder)
    except Exception as e:
        raise Exception(f'Failed to convert speech: {e}')

    # Save the resulting speech
    sf.write(output_audio_path, speech.numpy(), samplerate=16000)

    return output_audio_path

# test_function_code --------------------

from datasets import load_dataset

def test_convert_speech_with_different_voice():
    print("Testing started.")
    dataset = load_dataset('hf-internal-testing/librispeech_asr_demo', 'clean', split='validation')
    sample_data = dataset[0]['audio']['array']  # The sampled speech data for testing
    sample_rate = dataset.features['audio'].sampling_rate
    temp_input_audio_path = 'temp_input_audio.wav'
    temp_speaker_embedding_path = 'temp_xvector_speaker_embedding.npy'
    temp_output_audio_path = 'temp_converted_speech.wav'

    # Create temp files for the test
    sf.write(temp_input_audio_path, sample_data, sample_rate)
    np.save(temp_speaker_embedding_path, np.random.rand(512))  # Create a random speaker embedding

    # Testing case [1/1] started
    print("Testing case [1/1] started.")
    output_path = convert_speech_with_different_voice(temp_input_audio_path, temp_speaker_embedding_path, temp_output_audio_path)
    assert output_path == temp_output_audio_path, f"Test case [1/1] failed: Expected {temp_output_audio_path}, got {output_path}"  # Check if the function returns the correct output path

    # Clean up temp files
    import os
    os.remove(temp_input_audio_path)
    os.remove(temp_speaker_embedding_path)
    os.remove(temp_output_audio_path)

    print("Testing finished.")

# call_test_function_line --------------------

test_convert_speech_with_different_voice()