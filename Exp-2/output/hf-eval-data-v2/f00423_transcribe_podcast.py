# function_import --------------------

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

# function_code --------------------

def transcribe_podcast(podcast_file_path):
    """
    Transcribe a podcast file into text using the pre-trained 'jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli' model.

    Args:
        podcast_file_path (str): The path to the audio file to transcribe.

    Returns:
        str: The transcription of the audio file.
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(podcast_file_path)

    # Instantiate the ASR model and processor
    asr_model = Wav2Vec2ForCTC.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')
    asr_processor = Wav2Vec2Processor.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')

    # Process the audio file and return tensors
    input_tensor = asr_processor(waveform, return_tensors='pt').input_values

    # Get the logits from the model
    logits = asr_model(input_tensor).logits

    # Get the predictions from the logits
    predictions = torch.argmax(logits, dim=-1)

    # Decode the predictions into text
    transcription = asr_processor.batch_decode(predictions)[0]

    return transcription

# test_function_code --------------------

def test_transcribe_podcast():
    """
    Test the transcribe_podcast function with a sample audio file.
    """
    # Path to a sample audio file
    sample_audio_file_path = 'sample_audio.wav'

    # Call the function with the sample audio file
    transcription = transcribe_podcast(sample_audio_file_path)

    # Assert that the transcription is not empty
    assert transcription != ''

# call_test_function_code --------------------

test_transcribe_podcast()