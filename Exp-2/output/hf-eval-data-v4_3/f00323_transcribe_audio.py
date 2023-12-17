# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration


# function_code --------------------

def transcribe_audio(audio_data, audio_sampling_rate):
    """Transcribe the given audio data using a pretrained Whisper model.

    Args:
        audio_data (np.ndarray): The audio data to transcribe.
        audio_sampling_rate (int): The sampling rate of the audio data.

    Returns:
        str: The transcribed text of the audio data.

    Raises:
        ValueError: If the audio_data or audio_sampling_rate is invalid.
    """
    if not isinstance(audio_data, np.ndarray) or not isinstance(audio_sampling_rate, int):
        raise ValueError('Invalid audio data or sampling rate')

    processor = WhisperProcessor.from_pretrained('openai/whisper-large-v2')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v2')
    input_features = processor(audio_data, sampling_rate=audio_sampling_rate, return_tensors='pt').input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]


# test_function_code --------------------

def test_transcribe_audio():
    print('Testing started.')
    # Assume test_audio_data and test_audio_sampling_rate are predefined valid test inputs
    # Testing with valid input
    print('Testing case [1/2] started.')
    transcription = transcribe_audio(test_audio_data, test_audio_sampling_rate)
    assert isinstance(transcription, str), f'Test case [1/2] failed: Expected a string transcription, got {type(transcription)}'

    # Testing with invalid input
    print('Testing case [2/2] started.')
    try:
        transcribe_audio('invalid_audio_data', 'invalid_sampling_rate')
        assert False, 'Test case [2/2] failed: ValueError expected but not raised'
    except ValueError:
        assert True, 'Test case [2/2] succeeded: ValueError raised as expected'

    print('Testing finished.')


# call_test_function_line --------------------

test_transcribe_audio()