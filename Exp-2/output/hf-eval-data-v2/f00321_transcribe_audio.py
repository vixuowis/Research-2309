# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(audio_data):
    """
    Transcribe audio data into text using the Whisper ASR model.

    Args:
        audio_data (np.array): The audio data to be transcribed.

    Returns:
        transcription (str): The transcribed text.
    """
    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny.en')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny.en')
    input_features = processor(audio_data, sampling_rate=16000, return_tensors='pt').input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function with a sample from the LibriSpeech dataset.
    """
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = ds[0]['audio']
    transcription = transcribe_audio(sample)
    assert isinstance(transcription, str), 'The transcription should be a string.'

# call_test_function_code --------------------

test_transcribe_audio()