# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch

# function_code --------------------

def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes the audio from a given file using the Whisper model.

    Args:
        audio_file_path: A string representing the path to the audio file to be transcribed.

    Returns:
        A string containing the transcription of the audio.

    Raises:
        FileNotFoundError: If the audio file does not exist at the specified path.
        RuntimeError: If there is a problem processing the audio or generating the transcription.
    """
    # Check if the file exists
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f'Audio file not found at: {audio_file_path}')

    # Load the Whisper models
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')

    # Load the audio file, ensure the sampling rate is 16000 Hz
    sample = {'array': torch.load(audio_file_path), 'sampling_rate': 16000}
    input_features = processor(sample['array'], sampling_rate=sample['sampling_rate'], return_tensors='pt').input_features

    # Generate predictions and transcribe
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

# test_function_code --------------------

def test_transcribe_audio():
    print("Testing started.")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample_data = dataset[0]['audio']

    # Test case 1: Valid audio file
    print("Testing case [1/3] started.")
    transcription = transcribe_audio(sample_data['path'])
    assert transcription is not None, f"Test case [1/3] failed: Transcription should not be None"

    # Test case 2: Invalid path
    print("Testing case [2/3] started.")
    try:
        transcribe_audio('non_existent_file.wav')
        assert False, "Test case [2/3] failed: FileNotFoundError should have been raised"
    except FileNotFoundError as e:
        assert str(e) == "Audio file not found at: non_existent_file.wav", f"Test case [2/3] failed: {e}"

    # Test case 3: Empty file
    print("Testing case [3/3] started.")
    with open('empty_file.wav', 'wb') as f:
        pass
    try:
        transcribe_audio('empty_file.wav')
        assert False, "Test case [3/3] failed: RuntimeError should have been raised for an empty file"
    except RuntimeError as e:
        assert str(e) == "RuntimeError", f"Test case [3/3] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_transcribe_audio()