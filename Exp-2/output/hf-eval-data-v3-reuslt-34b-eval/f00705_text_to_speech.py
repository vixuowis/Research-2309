# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

# function_code --------------------

def text_to_speech(text: str, speaker_id: int = 7306):
    '''
    Converts a given text to speech using the SpeechT5 model from Hugging Face Transformers.

    Args:
        text (str): The text to be converted to speech.
        speaker_id (int, optional): The id of the speaker whose voice is to be used. Defaults to 7306.

    Returns:
        str: The path to the generated audio file.
    '''    
    processor = SpeechT5Processor.from_pretrained("google/speech-t5-large-ssv2-en-de")
    model = SpeechT5ForTextToSpeech.from_pretrained("google/speech-t5-large-ssv2-en-de")
    gan = SpeechT5HifiGan.from_pretrained("patrickvonplaten/speech-t5-hifigan-en-at")
    
    inputs = processor(text, return_tensors="pt", padding=True)
    
    input_features = inputs["input_features"]
    attention_mask = inputs["attention_mask"]
    speaker_ids = torch.tensor([[speaker_id]]).repeat(len(input_features), 1)
    
    hidden_states = model(
        input_features=input_features,
        attention_mask=attention_mask,
        speaker_ids=speaker_ids,
    )
    
    # Decoder has two outputs (mels and mel_lengths). Only the first one is important
    hidden_states = hidden_states[0]
    audio = gan(hidden_states).cpu()
    
    sf.write('audio.wav', audio, 16000)
    
    return 'audio.wav'

# test_function_code --------------------

def test_text_to_speech():
    '''
    Tests the text_to_speech function.
    '''
    assert text_to_speech('Hello, world!') == 'speech.wav'
    assert text_to_speech('This is a test.', 7306) == 'speech.wav'
    assert text_to_speech('Another test.', 7307) == 'speech.wav'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_text_to_speech()