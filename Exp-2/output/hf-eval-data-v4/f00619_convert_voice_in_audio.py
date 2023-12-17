# requirements_file --------------------

!pip install -U transformers datasets numpy torch soundfile

# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
import numpy as np
import soundfile as sf
import torch

# function_code --------------------

def convert_voice_in_audio(input_audio_path, output_audio_path, speaker_embedding_path):
    """
    Convert the voice in an audio file using a pretrained model without changing the content.

    Parameters:
    input_audio_path (str): Path to the input audio file.
    output_audio_path (str): Path where the converted audio will be saved.
    speaker_embedding_path (str): Path to the numpy file containing the speaker embeddings.

    Returns:
    None
    """

    # Load the input audio and sampling rate
    example_speech, sampling_rate = sf.read(input_audio_path)

    # Load the processor and models
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_vc')
    model = SpeechT5ForSpeechToSpeech.from_pretrained('microsoft/speecht5_vc')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

    # Preprocess the input audio
    inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors='pt')

    # Load speaker embeddings
    speaker_embeddings = np.load(speaker_embedding_path)
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

    # Generate the speech with the converted voice
    speech = model.generate_speech(inputs['input_values'], speaker_embeddings, vocoder=vocoder)

    # Save the resulting speech
    sf.write(output_audio_path, speech.numpy(), samplerate=16000)

# test_function_code --------------------

def test_convert_voice_in_audio():
    print("Testing voice conversion.")

    # Setup test data paths
    input_audio_path = 'input_audio.wav'
    output_audio_path = 'converted_speech.wav'
    speaker_embedding_path = 'xvector_speaker_embedding.npy'

    # Invoke the conversion function
    convert_voice_in_audio(input_audio_path, output_audio_path, speaker_embedding_path)

    # Read the converted speech file to ensure it was created
    converted_speech, converted_samplerate = sf.read(output_audio_path)
    assert converted_speech is not None, "Test failed: The converted speech file was not created."

    print("Test finished successfully.")