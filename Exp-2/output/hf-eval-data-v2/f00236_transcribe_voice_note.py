# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_voice_note(audio, sampling_rate):
    """
    Transcribe a voice note to text using the 'openai/whisper-large' model from Hugging Face Transformers.

    Args:
        audio (np.array): The voice note to be transcribed.
        sampling_rate (int): The sampling rate of the audio file.

    Returns:
        str: The transcribed text.
    """
    # Load the pre-trained 'openai/whisper-large' model and its processor
    processor = WhisperProcessor.from_pretrained('openai/whisper-large')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large')
    model.config.forced_decoder_ids = None

    # Convert the audio input into input features suitable for the model
    input_features = processor(audio, sampling_rate=sampling_rate, return_tensors='pt').input_features

    # Generate predicted IDs using the model
    predicted_ids = model.generate(input_features)

    # Decode the predicted IDs into text transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

    return transcription

# test_function_code --------------------

def test_transcribe_voice_note():
    """
    Test the 'transcribe_voice_note' function with a sample from the 'LibriSpeech (clean)' dataset.
    """
    # Load the 'LibriSpeech (clean)' dataset
    ds = load_dataset('librispeech_asr', 'clean', split='validation')

    # Select a sample from the dataset
    sample = ds[0]

    # Transcribe the voice note in the sample
    transcription = transcribe_voice_note(sample['audio'], sample['sampling_rate'])

    # Assert that the transcription is not empty
    assert transcription != ''

# call_test_function_code --------------------

test_transcribe_voice_note()