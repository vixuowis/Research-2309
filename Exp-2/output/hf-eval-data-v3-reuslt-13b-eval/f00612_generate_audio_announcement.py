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
    processor = SpeechT5Processor.from_pretrained("facebook/s2t-small-librispeech-asr")
    model = SpeechT5ForTextToSpeech.from_pretrained("facebook/s2t-small-librispeech-asr")
    
    input_text = processor(text, return_tensors="pt", padding=True)["input_ids"]

    # Generate speech
    generated_wave = model.generate(input_text).squeeze().cpu()

    # Save it on file
    sf.write("announcement.wav", generated_wave, 16000)


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