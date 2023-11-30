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
    
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny-random")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny-random").to('cuda').half()
    
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split="validation")
    processor.save_pretrained('processor')
    model.config.label_names = ['intent']
    model.config.return_dict = True    
    model.generate(**processor(ds['text'][0], return_tensors='pt'))
    model.to('cuda').half()

# -------------------- function_code --------------------

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