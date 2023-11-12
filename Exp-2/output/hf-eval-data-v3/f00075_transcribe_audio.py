# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(audio_sample):
    '''
    Transcribe audio using the openai/whisper-tiny model.

    Args:
        audio_sample (dict): A dictionary containing 'array' and 'sampling_rate' of the audio.

    Returns:
        str: The transcribed text.
    '''
    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny')
    input_features = processor(audio_sample['array'], sampling_rate=audio_sample['sampling_rate'], return_tensors='pt').input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

# test_function_code --------------------

def test_transcribe_audio():
    '''
    Test the transcribe_audio function.
    '''
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = ds[0]['audio']
    transcription = transcribe_audio(sample)
    assert isinstance(transcription, str), 'The result should be a string.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_transcribe_audio()