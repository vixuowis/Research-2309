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
    # load model and data collator
    processor = SpeechT5Processor.from_pretrained("csukuylab/speech-t5")
    speech_model = SpeechT5ForTextToSpeech.from_pretrained("csukuylab/speech-t5")
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
    
    # preprocess the text 
    input_ids = processor(text, return_tensors='pt', sampling_rate=24000).input_values
    batch = data_collator([{"input_ids": input_ids}])
        
    # generate speech
    with torch.no_grad():
        logits = speech_model(**batch, labels=input_ids).logits
    
    # save generated audio to file
    decoded_text = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
    wav = processor.parse_wav(decoded_text)
    path = f"{text}.wav"
    
    # synthesize speech
    with open(path, 'wb') as f:
        f.write(wav)        
    
    return path


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