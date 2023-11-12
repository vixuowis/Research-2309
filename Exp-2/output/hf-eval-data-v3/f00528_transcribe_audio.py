# function_import --------------------

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# function_code --------------------

def transcribe_audio(audio_file):
    """
    Transcribe an audio file into text using a pretrained ASR model.

    Args:
        audio_file (str): Path to the audio file to be transcribed.

    Returns:
        str: The transcribed text.

    Raises:
        OSError: If there is not enough disk space to download the model.
    """
    model = Wav2Vec2ForCTC.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')
    processor = Wav2Vec2Processor.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')
    # Load the audio file
    audio_input = processor(audio_file, return_tensors='pt', padding=True, truncation=True)
    # Generate the transcription
    logits = model(audio_input.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function with a sample audio file.
    """
    sample_audio_file = 'sample.wav'
    transcription = transcribe_audio(sample_audio_file)
    assert isinstance(transcription, str), 'The transcription should be a string.'
    assert len(transcription) > 0, 'The transcription should not be empty.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_transcribe_audio()