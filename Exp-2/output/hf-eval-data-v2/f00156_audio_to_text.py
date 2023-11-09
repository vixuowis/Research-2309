# function_import --------------------

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

# function_code --------------------

def audio_to_text(audio_file):
    """
    Transcribe an audio file to text using the Wav2Vec2ForCTC model from the Transformers library.

    Args:
        audio_file (str): Path to the audio file to transcribe.

    Returns:
        str: The transcribed text.
    """
    # Load the pre-trained processor and model
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

    # Load the audio file
    ds = load_dataset('patrickvonplaten/librispeech_asr_dummy', 'clean', split='validation')
    input_values = processor(ds[0]['audio']['array'], return_tensors='pt', padding='longest').input_values

    # Use the model to transcribe the audio to text
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the transcription
    transcription = processor.batch_decode(predicted_ids)

    return transcription

# test_function_code --------------------

def test_audio_to_text():
    """
    Test the audio_to_text function with a sample audio file.
    """
    # Define a sample audio file
    sample_audio_file = 'sample.wav'

    # Call the audio_to_text function
    transcription = audio_to_text(sample_audio_file)

    # Assert that the transcription is not empty
    assert transcription != ''

# call_test_function_code --------------------

test_audio_to_text()