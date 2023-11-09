# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_speech(audio_sample):
    """
    Transcribes spoken language into written text using the Whisper ASR model.

    Args:
        audio_sample (dict): A dictionary containing the audio data and sampling rate. The dictionary should have the following structure:
            {'array': numpy array, 'sampling_rate': int}

    Returns:
        str: The transcribed text.
    """
    # Initialize the Whisper processor and model
    processor = WhisperProcessor.from_pretrained('openai/whisper-base')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')

    # Preprocess the audio sample
    input_features = processor(audio_sample['array'], sampling_rate=audio_sample['sampling_rate'], return_tensors='pt').input_features

    # Generate the transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

# test_function_code --------------------

def test_transcribe_speech():
    """
    Tests the transcribe_speech function by transcribing a sample from the LibriSpeech dataset.
    """
    # Load the test dataset
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')

    # Select a sample from the dataset
    sample = ds[0]['audio']

    # Transcribe the sample
    transcription = transcribe_speech(sample)

    # Assert that the transcription is not empty
    assert transcription != ''

# call_test_function_code --------------------

test_transcribe_speech()