# requirements_file --------------------

import subprocess

requirements = ["transformers", "datasets"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio_to_text(audio: dict, sampling_rate: int) -> str:
    """
    Transcribes audio to text using the pre-trained Whisper model.

    Args:
        audio (dict): A dictionary containing the audio file data, with keys 'array' and 'sampling_rate'.
        sampling_rate (int): The sampling rate of the audio file.

    Returns:
        str: The transcribed text.

    Raises:
        ValueError: If 'audio' is not a dictionary or 'sampling_rate' is not an integer.
    """
    if not isinstance(audio, dict) or not isinstance(sampling_rate, int):
        raise ValueError("'audio' must be a dictionary and 'sampling_rate' must be an integer.")
    processor = WhisperProcessor.from_pretrained('openai/whisper-large')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large')
    model.config.forced_decoder_ids = None
    input_features = processor(audio['array'], sampling_rate=sampling_rate, return_tensors='pt').input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
    return transcription[0]

# test_function_code --------------------

def test_transcribe_audio_to_text():
    print("Testing started.")
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = ds[0]['audio']

    # Testing case 1: Valid audio sample
    print("Testing case [1/1] started.")
    result = transcribe_audio_to_text(sample, sample['sampling_rate'])
    assert isinstance(result, str), f"Test case [1/1] failed: Expected a string, but got {type(result)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_transcribe_audio_to_text()