# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(audio_sample):
    """
    Transcribe an audio sample using the Whisper ASR model.

    Args:
        audio_sample (dict): A dictionary containing the audio data and the sampling rate. The dictionary should have the following structure:
            {'array': numpy array, 'sampling_rate': int}

    Returns:
        str: The transcribed text.
    """
    # Load the Whisper processor and model
    processor = WhisperProcessor.from_pretrained('openai/whisper-base')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')

    # Preprocess the audio sample
    input_features = processor(audio_sample['array'], sampling_rate=audio_sample['sampling_rate'], return_tensors='pt').input_features

    # Generate the predicted IDs
    predicted_ids = model.generate(input_features)

    # Decode the predicted IDs to get the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function with a sample from the LibriSpeech dataset.
    """
    # Load the LibriSpeech dataset
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')

    # Get a sample from the dataset
    sample = ds[0]['audio']

    # Transcribe the audio sample
    transcription = transcribe_audio(sample)

    # Check that the transcription is a string
    assert isinstance(transcription, str)

# call_test_function_code --------------------

test_transcribe_audio()