# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio_samples(audio_samples):
    """
    Transcribes a collection of audio samples using the WhisperForConditionalGeneration model.

    Args:
        audio_samples (list): A list of dictionaries, each containing an 'array' key with the audio data and a 'sampling_rate' key with the sampling rate.

    Returns:
        list: A list of transcriptions for each audio sample.
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
    Tests the transcribe_audio_samples function by loading a sample dataset and comparing the output to expected results.
    """
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    audio_samples = [{'array': sample['audio'], 'sampling_rate': sample['sampling_rate']} for sample in ds]
    transcriptions = transcribe_audio_samples(audio_samples)
    assert len(transcriptions) == len(audio_samples), 'Number of transcriptions does not match number of audio samples.'
    for transcription in transcriptions:
        assert isinstance(transcription, str), 'Transcription is not a string.'

# call_test_function_code --------------------

test_transcribe_audio_samples()