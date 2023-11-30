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
    try:
        processor = SpeechT5Processor.from_pretrained("speech-t5-hifi-gan/v1")
        model = SpeechT5ForTextToSpeech.from_pretrained("speech-t5-hifi-gan/v1", vocab_size=processor.get_vocab_size())  # for PyTorch < 1.9.0 use "vocab_size" instead of "vocab_size_or_config_json_key"
        hifi_gan = SpeechT5HifiGan.from_pretrained("speech-t5-hifi-gan/v1")
        
        inputs = processor(text, return_tensors="pt", padding=True)
    
        with torch.no_grad():
            audio = model.generate(**inputs, num_beams=10, min_length=32000)
            
        audio = hifi_gan.batch_inference(audio)
        
        sf.write("./data/audio-announcement/output-audio.wav", audio[0].cpu().numpy(), 48000, "PCM_16")
    
    except:
        raise Exception('Error in generating audio announcement')

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