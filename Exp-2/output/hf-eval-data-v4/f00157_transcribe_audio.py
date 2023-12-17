# requirements_file --------------------

!pip install -U transformers datasets torch

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(audio_file_path):
    """
    Transcribe the given audio file using the Whisper model.

    Parameters:
        audio_file_path (str): The file path to the audio file to be transcribed.

    Returns:
        str: The transcribed text.

    """
    # Initialize the processor and model
    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny.en')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny.en')

    # Load and process the audio file
    with open(audio_file_path, 'rb') as audio_file:
        audio_input = {'array': audio_file.read(), 'sampling_rate': 16000}
    input_features = processor(audio_input['array'], sampling_rate=audio_input['sampling_rate'], return_tensors='pt').input_features

    # Generate the predicted token ids
    predicted_ids = model.generate(input_features)

    # Decode the predicted token ids into textual transcriptions
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

# test_function_code --------------------

def test_transcribe_audio():
    print("Testing transcribe_audio function.")

    # Test case 1: Check non-empty transcription result
    sample_audio = 'path/to/sample.wav'
    transcription = transcribe_audio(sample_audio)
    assert transcription, f"Transcription should not be empty."

    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_transcribe_audio()