# requirements_file --------------------

import subprocess

requirements = ["transformers", "datasets", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch

# function_code --------------------

def transcribe_audio(audio_path: str) -> str:
    '''
    Transcribes audio file to text using a pre-trained ASR model.

    Args:
        audio_path (str): The file path to the audio file that needs transcription.

    Returns:
        str: The transcribed text.

    Raises:
        FileNotFoundError: If the audio file is not found.
        Exception: If an error occurs during transcription.
    '''
    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny.en')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny.en')
    
    # Read the audio file
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    
    # Process and prepare features for the model
    input_features = processor(audio_data, sampling_rate=16000, return_tensors='pt').input_features
    
    # Transcribe audio
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription[0]


# test_function_code --------------------

def test_transcribe_audio():
    print("Testing started.")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample_data = dataset[0]  # extract a sample from the dataset
    audio_path = sample_data['file']

    # Test case 1: The audio file exists and can be transcribed
    print("Testing case [1/3] started.")
    transcription = transcribe_audio(audio_path)
    assert transcription, f"Test case [1/3] failed: Transcription should not be empty."

    # Test case 2: The audio file does not exist
    print("Testing case [2/3] started.")
    non_existing_audio_path = 'non_existing_file.wav'
    try:
        _ = transcribe_audio(non_existing_audio_path)
        assert False, "Test case [2/3] failed: Exception expected for non-existing file."
    except FileNotFoundError:
        pass  # Expected

    # Test case 3: Invalid audio file format is handled
    print("Testing case [3/3] started.")
    invalid_format_audio_path = 'invalid_format_file.txt'
    try:
        _ = transcribe_audio(invalid_format_audio_path)
        assert False, "Test case [3/3] failed: Exception expected for invalid audio format."
    except Exception as e:
        assert "Invalid data format" in str(e), f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# Run the test function
test_transcribe_audio()

# call_test_function_line --------------------

test_transcribe_audio()