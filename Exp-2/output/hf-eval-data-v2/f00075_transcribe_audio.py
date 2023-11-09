# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(audio_sample):
    """
    Transcribe audio data using the openai/whisper-tiny model.

    Args:
        audio_sample (dict): A dictionary containing 'array' and 'sampling_rate' of the audio.

    Returns:
        str: The transcribed text from the audio.
    """
    # Load the WhisperProcessor and model
    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny')

    # Process the raw audio data
    input_features = processor(audio_sample['array'], sampling_rate=audio_sample['sampling_rate'], return_tensors='pt').input_features

    # Use the model to generate a transcription
    predicted_ids = model.generate(input_features)

    # Decode the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function.
    """
    # Load the test dataset
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')

    # Select a sample from the dataset
    sample = ds[0]['audio']

    # Call the function with the sample
    result = transcribe_audio(sample)

    # Assert that the result is a string
    assert isinstance(result, str)

# call_test_function_code --------------------

test_transcribe_audio()