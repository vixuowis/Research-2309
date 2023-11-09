# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import numpy as np
import torch
import soundfile as sf

# function_code --------------------

def change_speaker_voice(audio, sampling_rate, speaker_embeddings_file):
    '''
    Change the speaker's voice in a recorded podcast using the SpeechT5 model.
    
    Args:
        audio (numpy array): The audio data to be processed.
        sampling_rate (int): The sampling rate of the audio data.
        speaker_embeddings_file (str): The path to the speaker embeddings file.
    
    Returns:
        None. The output audio is saved to a file named 'speech.wav'.
    '''
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_vc')
    model = SpeechT5ForSpeechToSpeech.from_pretrained('microsoft/speecht5_vc')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')
    inputs = processor(audio=audio, sampling_rate=sampling_rate, return_tensors='pt')
    speaker_embeddings = np.load(speaker_embeddings_file)
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
    speech = model.generate_speech(inputs['input_values'], speaker_embeddings, vocoder=vocoder)
    sf.write('speech.wav', speech.numpy(), samplerate=16000)

# test_function_code --------------------

def test_change_speaker_voice():
    '''
    Test the change_speaker_voice function.
    
    This function does not return anything. It raises an error if the change_speaker_voice function
    does not work as expected.
    '''
    dataset = load_dataset('hf-internal-testing/librispeech_asr_demo', 'clean', split='validation')
    example_speech = dataset[0]['audio']['array']
    sampling_rate = dataset.features['audio'].sampling_rate
    speaker_embeddings_file = 'xvector_speaker_embedding.npy'
    try:
        change_speaker_voice(example_speech, sampling_rate, speaker_embeddings_file)
    except Exception as e:
        print(f'change_speaker_voice function failed with error {e}')
        assert False

# call_test_function_code --------------------

test_change_speaker_voice()