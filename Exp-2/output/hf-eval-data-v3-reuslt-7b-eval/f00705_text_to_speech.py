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
    
    # load model and processor -----
    processor = SpeechT5Processor.from_pretrained('facebook/s2t-small-librispeech-asr')
    model = SpeechT5ForTextToSpeech.from_pretrained("facebook/s2t-small-librispeech-asr", 
                                                    processor=processor, 
                                                    speaker_set=[str(i) for i in range(9008)]).to('cuda')
    
    # tokenize -----
    input_ids = processor.batch_encode_plus([text], return_tensors='pt').input_ids
    
    with torch.no_grad():
        output = model(input_ids, 
                       speaker_id=speaker_id)
        
    # save audio -----
    sf.write('generated-audio.wav', 
             output[0].cpu().numpy(), 
             24000)
    
    return 'generated-audio.wav'

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