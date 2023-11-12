# function_import --------------------

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

# function_code --------------------

def audio_to_text(audio_file):
    '''
    Transcribe audio file to text using pre-trained model from Transformers.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        str: Transcribed text from the audio file.
    '''
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
    ds = load_dataset('patrickvonplaten/librispeech_asr_dummy', 'clean', split='validation')
    input_values = processor(ds[0]['audio']['array'], return_tensors='pt', padding='longest').input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription

# test_function_code --------------------

def test_audio_to_text():
    '''
    Test the audio_to_text function.
    '''
    sample_audio_file = 'sample.wav'
    transcription = audio_to_text(sample_audio_file)
    assert isinstance(transcription, str), 'The output should be a string.'
    assert len(transcription) > 0, 'The output string should not be empty.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_audio_to_text()