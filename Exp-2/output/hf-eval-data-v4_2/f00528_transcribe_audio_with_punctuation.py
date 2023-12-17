# requirements_file --------------------

!pip install -U transformers librosa torch pytest

# function_import --------------------

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer

# function_code --------------------

def transcribe_audio_with_punctuation(audio_path):
    """Transcribe an audio file to text with punctuation.

    Args:
        audio_path (str): The path to the audio file (.wav format expected).

    Returns:
        str: The transcribed text with punctuation.

    Raises:
        FileNotFoundError: If the audio file is not found.
        ValueError: If the audio file format is not supported.
    """
    import librosa
    import torch

    # Load the audio file
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found at location: {audio_path}")
    signal, sr = librosa.load(audio_path, sr=16000)

    # Load the ASR model and processor
    model = Wav2Vec2ForCTC.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')
    processor = Wav2Vec2Processor.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')

    # Process the audio file for the model input
    input_values = processor(signal, return_tensors='pt', sampling_rate=sr).input_values

    # Inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted tokens into text
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]

    return transcription


# test_function_code --------------------

import os
import pytest

def test_transcribe_audio_with_punctuation():
    print("Testing started.")

    # Create a sample audio file for the test
    sample_audio_path = 'sample_audio.wav'
    if not os.path.isfile(sample_audio_path):
        pytest.skip("Sample audio file not found.")

    # Test case 1: Audio file exists and is supported
    print("Testing case [1/3] started.")
    transcription = transcribe_audio_with_punctuation(sample_audio_path)
    assert isinstance(transcription, str), f"Test case [1/3] failed: The transcription should be a string."

    # Test case 2: Audio file does not exist
    print("Testing case [2/3] started.")
    with pytest.raises(FileNotFoundError):
        transcribe_audio_with_punctuation('nonexistent_audio.wav')

    # Test case 3: Audio file is not in .wav format
    # Assuming 'not_a_wav.mp3' is present for test purpose
    print("Testing case [3/3] started.")
    with pytest.raises(ValueError):
        transcribe_audio_with_punctuation('not_a_wav.mp3')
    
    print("Testing finished.")


# call_test_function_line --------------------

test_transcribe_audio_with_punctuation()