# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(audio_path):
    """
    Transcribe the provided audio file using the Whisper ASR model.

    :param audio_path: Path to the audio file to be transcribed.
    :return: Transcription result as a string.
    """
    # Initialize the processor and model
    processor = WhisperProcessor.from_pretrained('openai/whisper-large-v2')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v2')

    # Load audio data
    with open(audio_path, 'rb') as audio_file:
        audio_data = audio_file.read()

    # Process the audio file and generate input features
    input_features = processor(audio_data, sampling_rate=16000, return_tensors='pt').input_features

    # Generate predictions
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    print("Testing transcribe_audio function.")
    # Load a sample from the dataset
    dataset = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = dataset[0]['audio']
    audio_path = sample['file']

    # Test case
    transcription = transcribe_audio(audio_path)
    assert isinstance(transcription, str), "The result should be a string."
    assert len(transcription) > 0, "The transcription should not be empty."
    print("Test successful!")

# Run the test
test_transcribe_audio()