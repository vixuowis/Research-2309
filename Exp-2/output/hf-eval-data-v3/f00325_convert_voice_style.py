# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
import soundfile as sf
import torch
import numpy as np
import os

# function_code --------------------

def convert_voice_style(audio_file_path: str, speaker_embedding_path: str, output_file_path: str) -> None:
    '''
    Convert the voice style of a given audio file using a specified speaker's embeddings.

    Args:
        audio_file_path (str): The path to the audio file to be converted.
        speaker_embedding_path (str): The path to the numpy file containing the speaker's embeddings.
        output_file_path (str): The path to save the converted audio file.

    Returns:
        None
    '''
    # Load the audio file
    example_speech, sampling_rate = sf.read(audio_file_path)

    # Initialize the processor and models
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_vc')
    model = SpeechT5ForSpeechToSpeech.from_pretrained('microsoft/speecht5_vc')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

    # Process the audio
    inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors='pt')

    # Load the speaker's embeddings
    speaker_embeddings = np.load(speaker_embedding_path)
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

    # Generate the converted speech
    speech = model.generate_speech(inputs['input_values'], speaker_embeddings, vocoder=vocoder)

    # Save the converted speech to a file
    sf.write(output_file_path, speech.numpy(), samplerate=16000)

# test_function_code --------------------

def test_convert_voice_style():
    '''
    Test the convert_voice_style function.
    '''
    # Define the paths to the test audio file and speaker's embeddings
    audio_file_path = 'test_audio.wav'
    speaker_embedding_path = 'test_speaker_embedding.npy'
    output_file_path = 'test_output.wav'

    # Call the function
    convert_voice_style(audio_file_path, speaker_embedding_path, output_file_path)

    # Check that the output file was created
    assert os.path.exists(output_file_path), 'Output file was not created.'

    # Check that the output file is not empty
    assert os.path.getsize(output_file_path) > 0, 'Output file is empty.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_convert_voice_style()