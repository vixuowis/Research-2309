# function_import --------------------

import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# function_code --------------------

def transcribe_audio(audio_data, audio_sampling_rate):
    """
    Transcribe audio data using the Whisper ASR model from Hugging Face Transformers.

    Args:
        audio_data (np.array): The audio data to be transcribed.
        audio_sampling_rate (int): The sampling rate of the audio data.

    Returns:
        str: The transcription of the audio data.
    """
    processor = WhisperProcessor.from_pretrained('openai/whisper-large-v2')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v2')
    input_features = processor(audio_data, sampling_rate=audio_sampling_rate, return_tensors='pt').input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function.
    """
    # Test case: short audio clip
    audio_data = np.random.rand(16000)
    audio_sampling_rate = 16000
    transcription = transcribe_audio(audio_data, audio_sampling_rate)
    assert isinstance(transcription, str), 'The transcription should be a string.'

    # Test case: long audio clip
    audio_data = np.random.rand(160000)
    audio_sampling_rate = 16000
    transcription = transcribe_audio(audio_data, audio_sampling_rate)
    assert isinstance(transcription, str), 'The transcription should be a string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_transcribe_audio()