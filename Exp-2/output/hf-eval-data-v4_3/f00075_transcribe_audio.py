# requirements_file --------------------

import subprocess

requirements = ["transformers", "datasets"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(audio, sampling_rate):
    """
    Transcribes the given audio input while preserving the accent or language.

    Args:
        audio (np.ndarray): The raw audio waveform as a numpy array.
        sampling_rate (int): The sampling rate of the audio input.

    Returns:
        str: The transcription of the audio input.

    Raises:
        ValueError: If audio is None or sampling_rate is not positive.
    """
    if audio is None or sampling_rate <= 0:
        raise ValueError('Invalid audio or sampling rate.')

    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny')

    input_features = processor(audio, sampling_rate=sampling_rate, return_tensors='pt').input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

# test_function_code --------------------

def test_transcribe_audio():
    print("Testing started.")
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = ds[0]['audio']

    # Test case 1: Valid audio data
    print("Testing case [1/1] started.")
    transcription = transcribe_audio(sample['array'], sample['sampling_rate'])
    assert isinstance(transcription, str), f"Test case [1/1] failed: The transcription should be a string."
    print("Testing finished.")

# call_test_function_line --------------------

test_transcribe_audio()