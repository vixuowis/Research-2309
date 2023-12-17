# requirements_file --------------------

!pip install -U transformers datasets torch

# function_import --------------------

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

# function_code --------------------

def transcribe_podcast(audio_file_path):
    """
    Transcribes a podcast audio file to text using the Wav2Vec2 model.

    :param audio_file_path: Path to the podcast audio file (wav format)
    :return: Transcription of the audio file
    """
    # Load pretrained models
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

    # Load audio file
    audio_input, _ = load_dataset('audio', data_files=[audio_file_path], split='train')[0]['audio']

    # Process audio to input values
    input_values = processor(audio_input, return_tensors='pt', padding='longest').input_values

    # Transcribe audio
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

# test_function_code --------------------

def test_transcribe_podcast():
    print("Testing transcribe_podcast function.")
    audio_file_path = 'dummy_audio.wav'  # Assume this is a valid audio file path for the test

    # Test case 1: Check if the transcription is a string
    print("Testing case [1/1] started.")
    transcription = transcribe_podcast(audio_file_path)
    assert isinstance(transcription, str), f"Test case [1/1] failed: Expected a string, got {type(transcription)}"
    print("Test case [1/1] passed.")
    print("Testing finished.")

    # In a real-world scenario, additional test cases would compare the transcription to a known ground truth.

# Run the test function
test_transcribe_podcast()