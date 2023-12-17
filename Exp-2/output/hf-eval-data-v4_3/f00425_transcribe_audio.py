# requirements_file --------------------

import subprocess

requirements = ["transformers", "datasets"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes the given audio file into text using the Whisper model.

    Args:
        file_path (str): The path to the audio file to be transcribed.

    Returns:
        str: The transcribed text.

    Raises:
        FileNotFoundError: If the file_path does not exist.
        Exception: If transcription fails for other reasons.
    """
    import os
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')

    with open(file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
    
    # Would require actual audio sample rate
    sample_rate = 16000

    input_features = processor(
        {'array': audio_data, 'sampling_rate': sample_rate},
        return_tensors='pt'
    ).input_features

    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

# test_function_code --------------------

def test_transcribe_audio():
    print("Testing started.")
    dataset = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample_data = dataset[0]['audio']
    assert transcribe_audio(sample_data['path']) == sample_data['text'], f"Test failed: Audio transcription does not match expected text."
    print("Testing finished.")

# call_test_function_line --------------------

test_transcribe_audio()