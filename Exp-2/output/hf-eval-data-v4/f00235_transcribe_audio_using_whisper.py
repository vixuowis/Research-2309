# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio_using_whisper(audio_path: str) -> str:
    """
    Transcribes the content of an audio file using OpenAI's Whisper ASR model.

    :param audio_path: Path to the audio file to be transcribed.
    :return: The transcribed text from the audio.
    """
    # Initialize the Whisper processor and model with the base variant
    processor = WhisperProcessor.from_pretrained('openai/whisper-base')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')

    # Load the audio file
    with open(audio_path, 'rb') as f:
        audio_data = f.read()

    # Preprocess the audio
    input_features = processor(audio_data, return_tensors='pt', sampling_rate=16000).input_features

    # Generate the transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription

# test_function_code --------------------

def test_transcribe_audio_using_whisper():
    print("Testing transcribe_audio_using_whisper function.")

    # Test case 1: Known audio content
    print("Testing case [1/1] started.")
    test_audio_path = 'test_audio.wav' # Replace with actual test audio path
    expected_transcription = 'Hello world' # Replace with expected transcription of the test audio
    transcription = transcribe_audio_using_whisper(test_audio_path)
    assert transcription == expected_transcription, f"Test case [1/1] failed: Expected '{{expected_transcription}}' but got '{{transcription}}'"
    print("Testing case [1/1] finished successfully.")

    print("Testing finished.")

test_transcribe_audio_using_whisper()