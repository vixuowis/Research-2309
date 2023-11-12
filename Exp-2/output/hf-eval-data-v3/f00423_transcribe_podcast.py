# function_import --------------------

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

# function_code --------------------

def transcribe_podcast(podcast_file_path):
    """
    Transcribe a podcast from a given file path using a pre-trained ASR model.

    Args:
        podcast_file_path (str): The file path of the podcast to be transcribed.

    Returns:
        str: The transcription of the podcast.

    Raises:
        FileNotFoundError: If the podcast file does not exist.
        RuntimeError: If there is an error during the transcription process.
    """
    asr_model = Wav2Vec2ForCTC.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')
    asr_processor = Wav2Vec2Processor.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')
    waveform, sample_rate = torchaudio.load(podcast_file_path)
    input_values = asr_processor(waveform, return_tensors='pt').input_values
    logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.decode(predicted_ids[0])
    return transcription

# test_function_code --------------------

def test_transcribe_podcast():
    """
    Test the transcribe_podcast function.
    """
    # Test with a sample podcast file
    try:
        transcription = transcribe_podcast('sample_podcast.wav')
        assert isinstance(transcription, str)
        print('Test passed!')
    except Exception as e:
        print('Test failed:', e)

test_transcribe_podcast()

# call_test_function_code --------------------

test_transcribe_podcast()