# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_speech_to_text(audio_sample, sampling_rate):
    # Initialize Whisper processor and model with the pretrained weights
    processor = WhisperProcessor.from_pretrained('openai/whisper-base')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')

    # Process the audio sample to generate input features for the model
    input_features = processor(audio_sample, sampling_rate=sampling_rate, return_tensors='pt').input_features

    # Generate the predicted ids (transcription) from the model
    predicted_ids = model.generate(input_features)

    # Decode the predicted ids into text transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0] # Return the first transcription (assuming one sample)

# test_function_code --------------------

def test_transcribe_speech_to_text():
    print('Testing transcribe_speech_to_text function.')
    # Load a sample from the dataset
    dataset = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = dataset[0]['audio']

    # Test case: Transcribe the first audio sample in the dataset
    transcription = transcribe_speech_to_text(sample['array'], sample['sampling_rate'])
    expected_transcription = 'MR. QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'
    assert transcription.upper().strip() == expected_transcription, f'Test failed: Expected transcription does not match the actual.\nExpected: {expected_transcription}\nActual: {transcription}'

    print('All tests passed!')

test_transcribe_speech_to_text()