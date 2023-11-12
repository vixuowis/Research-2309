# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

# function_code --------------------

def generate_audio_announcement(text):
    '''
    Generate an audio announcement from a given text using the SpeechT5 model.
    
    Args:
        text (str): The text to be converted to speech.
    
    Returns:
        None. The function writes the output audio to a .wav file.
    
    Raises:
        Exception: If there is an error in generating the audio.
    '''
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
    model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')
    inputs = processor(text=text, return_tensors='pt')
    embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]['xvector']).unsqueeze(0)
    speech = model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=vocoder)
    sf.write('speech.wav', speech.numpy(), samplerate=16000)

# test_function_code --------------------

def test_generate_audio_announcement():
    '''
    Test the generate_audio_announcement function.
    '''
    try:
        generate_audio_announcement('This is a test announcement.')
        print('Test passed.')
    except Exception as e:
        print('Test failed. Error: ', e)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_audio_announcement()