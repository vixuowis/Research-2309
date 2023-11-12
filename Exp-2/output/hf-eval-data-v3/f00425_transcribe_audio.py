# function_import --------------------

import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(audio_sample, sampling_rate):
    '''
    Transcribe an audio file using the Whisper model from Hugging Face Transformers.

    Args:
        audio_sample (np.array): The audio data to be transcribed.
        sampling_rate (int): The sampling rate of the audio data.

    Returns:
        str: The transcribed text.
    '''
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')
    model.config.forced_decoder_ids = None

    input_features = processor(audio_sample, sampling_rate=sampling_rate, return_tensors='pt').input_features

    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    '''
    Test the transcribe_audio function.
    '''
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = ds[0]['audio']
    sampling_rate = ds[0]['sampling_rate']

    transcription = transcribe_audio(sample, sampling_rate)
    assert isinstance(transcription, str), 'The transcription should be a string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_transcribe_audio()