# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio_file(audio_file_path):
    # Instantiate a Whisper processor and model
    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')

    # Load the audio data and preprocess
    audio_data = load_dataset('load_audio', data_files=audio_file_path)['train'][0]['audio']
    input_features = processor(audio_data['array'], sampling_rate=audio_data['sampling_rate'], return_tensors='pt').input_features

    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

# test_function_code --------------------

def test_transcribe_audio_file():
    print("Testing audio transcription started.")
    dataset = load_dataset("superb_dummy", "asr", split='validation')
    sample_data = dataset[0]

    # Test for the correct transcription of a sample data
    print("Testing sample data transcription [1/1] started.")
    output = transcribe_audio_file(sample_data['file'])
    assert isinstance(output, list) and isinstance(output[0], str), f"Test failed: The transcription should be a list of strings."
    print("Sample data transcription test finished.")

    print("Testing audio transcription finished.")

# Run the test function
test_transcribe_audio_file()