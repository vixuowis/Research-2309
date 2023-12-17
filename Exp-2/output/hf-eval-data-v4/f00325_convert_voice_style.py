# requirements_file --------------------

!pip install -U transformers datasets numpy torch soundfile

# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
import soundfile as sf
import torch
import numpy as np

# function_code --------------------

def convert_voice_style(input_audio_path, speaker_embedding_path, output_audio_path='converted_speech.wav', sampling_rate=16000):
    """
    Convert the style of an input audio file to the voice of a target speaker using a pre-trained voice conversion model.

    :param input_audio_path: Path to the input audio file.
    :param speaker_embedding_path: Path to the numpy file containing target speaker embeddings.
    :param output_audio_path: Path to save the converted audio file.
    :param sampling_rate: Sampling rate for audio processing.
    :return: None
    """
    # Load audio
    original_audio, _ = sf.read(input_audio_path)

    # Initialize processor and models
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_vc')
    model = SpeechT5ForSpeechToSpeech.from_pretrained('microsoft/speecht5_vc')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

    # Prepare input features
    inputs = processor(audio=original_audio, sampling_rate=sampling_rate, return_tensors='pt')

    # Load speaker embeddings
    speaker_embeddings = np.load(speaker_embedding_path)
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

    # Generate converted speech
    with torch.no_grad():
        converted_speech = model.generate_speech(inputs['input_values'], speaker_embeddings, vocoder=vocoder)

    # Save the converted speech to a file
    sf.write(output_audio_path, converted_speech.numpy(), samplerate=sampling_rate)



# test_function_code --------------------

def test_convert_voice_style():
    print("Testing started.")

    # Assuming there's a test audio and embedding available
    input_audio_path = 'test_input.wav'
    speaker_embedding_path = 'test_speaker_embedding.npy'
    output_audio_path = 'test_output.wav'

    # Test case 1: Check if function runs without errors
    print("Testing case [1/2] started.")
    try:
        convert_voice_style(input_audio_path, speaker_embedding_path, output_audio_path)
        assertion = True
    except Exception as e:
        print(f"Test case [1/2] failed: {e}")
        assertion = False
    assert assertion, "Test case [1/2] failed: Function encounter an error."

    # Test case 2: Check if output file exists after running the function
    print("Testing case [2/2] started.")
    assert os.path.exists(output_audio_path), f"Test case [2/2] failed: Output audio file {output_audio_path} does not exist."
    print("Testing finished.")

# Run the test
import os
test_convert_voice_style()
