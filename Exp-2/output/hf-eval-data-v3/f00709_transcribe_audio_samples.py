# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import soundfile as sf

# function_code --------------------

def transcribe_audio_samples(audio_samples):
    """
    Transcribe a collection of audio samples into text using the WhisperForConditionalGeneration model.

    Args:
        audio_samples (list): A list of dictionaries, each containing an 'array' key with the audio data and a 'sampling_rate' key with the sampling rate.

    Returns:
        list: A list of transcriptions for each audio sample.

    Raises:
        ImportError: If the required 'soundfile' module is not installed.
    """
    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')

    transcriptions = []
    for audio_sample in audio_samples:
        input_features = processor(audio_sample['array'], sampling_rate=audio_sample['sampling_rate'], return_tensors='pt').input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcriptions.append(transcription)

    return transcriptions

# test_function_code --------------------

def test_transcribe_audio_samples():
    """
    Test the transcribe_audio_samples function with a dummy dataset.
    """
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    audio_samples = [{'array': sample['audio'], 'sampling_rate': sample['sampling_rate']} for sample in ds]
    transcriptions = transcribe_audio_samples(audio_samples)
    assert isinstance(transcriptions, list), 'The result should be a list.'
    assert len(transcriptions) == len(audio_samples), 'The number of transcriptions should be equal to the number of audio samples.'
    assert all(isinstance(t, str) for t in transcriptions), 'Each transcription should be a string.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_transcribe_audio_samples()