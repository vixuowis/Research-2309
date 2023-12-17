# requirements_file --------------------

import subprocess

requirements = ["transformers", "librosa", "datasets"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# function_code --------------------

def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes the given audio file to text using the Whisper model.

    Args:
        audio_file_path: A string with the path to the audio file to transcribe.

    Returns:
        A string containing the transcribed text.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the file format is unsupported.
    """
    # Load the pre-trained Whisper model
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')

    # Load the audio file with librosa
    try:
        audio_data, sampling_rate = librosa.load(audio_file_path, sr=None)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {audio_file_path} was not found.")
    except ValueError as e:
        raise ValueError(f"Error loading file {audio_file_path}: {e}")

    # Process the audio data
    input_features = processor(audio_data, sampling_rate=sampling_rate, return_tensors='pt').input_features

    # Generate the transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription

# test_function_code --------------------

from datasets import load_dataset

def test_transcribe_audio():
    print("Testing started.")
    # Load a dummy audio dataset as an example
    dataset = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample_data = dataset[0]['audio']

    # Testing case 1: A valid audio file
    print("Testing case [1/1] started.")
    transcription = transcribe_audio(sample_data['path'])
    assert isinstance(transcription, str), f"Test case [1/1] failed: The transcription should be a string."
    print("Testing finished.")

# call_test_function_line --------------------

test_transcribe_audio()