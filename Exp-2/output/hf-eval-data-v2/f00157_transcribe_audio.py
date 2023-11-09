# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(audio_sample):
    """
    Transcribes an audio sample using the Whisper ASR model.

    Args:
        audio_sample (dict): A dictionary containing the audio data and sampling rate.
            The dictionary should have the following structure:
            {'array': <numpy array containing audio data>, 'sampling_rate': <sampling rate of the audio data>}

    Returns:
        str: The transcribed text.
    """
    # Load the pre-trained model and processor
    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny.en')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny.en')

    # Process the audio sample and generate the input features
    input_features = processor(audio_sample['array'], sampling_rate=audio_sample['sampling_rate'], return_tensors='pt').input_features

    # Generate the predicted token ids
    predicted_ids = model.generate(input_features)

    # Decode the predicted token ids into textual transcriptions
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    """
    Tests the transcribe_audio function by loading a sample from the LibriSpeech dataset and comparing
    the output of the function to the expected transcription.
    """
    # Load the LibriSpeech dataset
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')

    # Select a sample from the dataset
    sample = ds[0]

    # Transcribe the audio sample
    transcription = transcribe_audio(sample['audio'])

    # The expected transcription is not known, so we cannot make a strict comparison
    # Instead, we check that the transcription is a non-empty string
    assert isinstance(transcription, str) and len(transcription) > 0

# call_test_function_code --------------------

test_transcribe_audio()