# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

# function_code --------------------

def text_to_speech(text):
    '''
    This function converts text to speech using the SpeechT5 model from Hugging Face Transformers.
    
    Args:
        text (str): The text to be converted to speech.
    
    Returns:
        None. The function writes the generated speech to an audio file named 'speech.wav'.
    
    Raises:
        Exception: If there is an error in loading the models or generating the speech.
    '''
    try:
        processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
        model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')
        vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')
        inputs = processor(text=text, return_tensors='pt')
        embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]['xvector']).unsqueeze(0)
        speech = model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=vocoder)
        sf.write('speech.wav', speech.numpy(), samplerate=16000)
    except Exception as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_text_to_speech():
    '''
    This function tests the text_to_speech function by passing a sample text and checking if the output audio file is created.
    '''
    import os
    text = 'Hello, my dog is cute'
    text_to_speech(text)
    assert os.path.exists('speech.wav'), 'The audio file was not created.'

# call_test_function_code --------------------

test_text_to_speech()