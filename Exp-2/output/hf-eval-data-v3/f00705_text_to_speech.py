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
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
    model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')
    inputs = processor(text=text, return_tensors='pt')
    embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
    speaker_embeddings = torch.tensor(embeddings_dataset[speaker_id]['xvector']).unsqueeze(0)
    speech = model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=vocoder)
    sf.write('speech.wav', speech.numpy(), samplerate=16000)
    return 'speech.wav'

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