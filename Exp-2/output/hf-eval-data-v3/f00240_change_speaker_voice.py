# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import numpy as np
import torch
import soundfile as sf

# function_code --------------------

def change_speaker_voice(audio_file: str, speaker_embedding_file: str, output_file: str = 'speech.wav'):
    '''
    Change the speaker's voice in a recorded podcast using Hugging Face Transformers.

    Args:
        audio_file (str): Path to the audio file.
        speaker_embedding_file (str): Path to the speaker embedding file.
        output_file (str, optional): Path to the output file. Defaults to 'speech.wav'.

    Returns:
        None
    '''
    dataset = load_dataset('hf-internal-testing/librispeech_asr_demo', 'clean', split='validation')
    example_speech = dataset[0]['audio']['array']
    sampling_rate = dataset.features['audio'].sampling_rate
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_vc')
    model = SpeechT5ForSpeechToSpeech.from_pretrained('microsoft/speecht5_vc')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

    inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors='pt')
    speaker_embeddings = np.load(speaker_embedding_file)
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
    speech = model.generate_speech(inputs['input_values'], speaker_embeddings, vocoder=vocoder)

    sf.write(output_file, speech.numpy(), samplerate=16000)

# test_function_code --------------------

def test_change_speaker_voice():
    '''
    Test the function change_speaker_voice.
    '''
    # Test case 1
    try:
        change_speaker_voice('audio1.wav', 'speaker1.npy', 'output1.wav')
    except Exception as e:
        assert str(e) == 'File audio1.wav not found.'

    # Test case 2
    try:
        change_speaker_voice('audio2.wav', 'speaker2.npy', 'output2.wav')
    except Exception as e:
        assert str(e) == 'File speaker2.npy not found.'

    # Test case 3
    try:
        change_speaker_voice('audio3.wav', 'speaker3.npy', 'output3.wav')
    except Exception as e:
        assert str(e) == 'File output3.wav not found.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_change_speaker_voice()