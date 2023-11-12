# function_import --------------------

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_speech(audio_sample):
    """
    Transcribe spoken language into written text using the Whisper ASR model.

    Args:
        audio_sample (dict): A dictionary containing the audio data and its sampling rate.
            The dictionary has the following structure:
            {
                'array': The audio data as a numpy array,
                'sampling_rate': The sampling rate of the audio data
            }

    Returns:
        str: The transcribed text.
    """
    processor = WhisperProcessor.from_pretrained('openai/whisper-base')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')

    input_features = processor(audio_sample['array'], sampling_rate=audio_sample['sampling_rate'], return_tensors='pt').input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

# test_function_code --------------------

def test_transcribe_speech():
    """
    Test the transcribe_speech function.
    """
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = ds[0]['audio']

    transcription = transcribe_speech(sample)
    assert isinstance(transcription, str), 'The transcription should be a string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_transcribe_speech()