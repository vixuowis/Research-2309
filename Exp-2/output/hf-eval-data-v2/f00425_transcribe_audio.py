# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(sample, sampling_rate):
    '''
    Transcribe an audio sample using the Whisper model from Hugging Face Transformers.

    Args:
        sample (np.array): The audio sample to transcribe.
        sampling_rate (int): The sampling rate of the audio sample.

    Returns:
        str: The transcription of the audio sample.
    '''
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')
    model.config.forced_decoder_ids = None

    input_features = processor(sample, sampling_rate=sampling_rate, return_tensors='pt').input_features

    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    '''
    Test the transcribe_audio function.
    '''
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = ds[0]['audio']
    sampling_rate = ds[0]['sampling_rate']

    transcription = transcribe_audio(sample, sampling_rate)
    assert isinstance(transcription, str), 'The transcription should be a string.'

# call_test_function_code --------------------

test_transcribe_audio()