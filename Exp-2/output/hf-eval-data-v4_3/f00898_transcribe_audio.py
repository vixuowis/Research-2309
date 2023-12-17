# requirements_file --------------------

import subprocess

requirements = ["transformers", "datasets", "torch", "librosa"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

# function_code --------------------

def transcribe_audio(audio_file):
    """Converts speech from an audio file to text using the Wav2Vec2ForCTC model from Hugging Face.

    Args:
        audio_file (str): The file path of the audio file to be transcribed.

    Returns:
        str: The transcribed text.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio file is not in the correct format.
    """
    # Load the processor and model
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')

    # Load audio file
    audio_input, _ = librosa.load(audio_file, sr=16000)

    # Preprocess audio input
    input_values = processor(audio_input, return_tensors='pt', padding='longest').input_values

    # Retrieve logits from the model
    logits = model(input_values).logits

    # Predict the IDs from logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the IDs to text
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]

# test_function_code --------------------

def test_transcribe_audio():
    print("Testing started.")
    # Prepare the sample audio dataset for testing
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    sample_data = dataset[0]

    # Testing case 1
    print("Testing case [1/1] started.")
    transcription = transcribe_audio(sample_data['file'])
    expected_transcription = sample_data['text']
    assert transcription == expected_transcription, f"Test case [1/1] failed: expected '{{expected_transcription}}', got '{{transcription}}'"
    print("Testing finished.")

# call_test_function_line --------------------

test_transcribe_audio()