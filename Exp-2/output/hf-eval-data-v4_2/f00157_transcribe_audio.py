# requirements_file --------------------

!pip install -U transformers datasets torch torchaudio

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe the given audio file into text using the openai/whisper-tiny.en model.

    Args:
        audio_path: The path to the audio file to transcribe.

    Returns:
        The transcribed text as a string.

    Raises:
        FileNotFoundError: If the audio_path does not exist.
        Exception: If any other error during transcription.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f'Audio file not found: {audio_path}')
    
    try:
        # Load the required model and processor
        model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny.en')
        processor = WhisperProcessor.from_pretrained('openai/whisper-tiny.en')
        
        # Load and process the audio file
        audio_input, sampling_rate = torchaudio.load(audio_path)
        input_features = processor(audio_input, sampling_rate=sampling_rate, return_tensors='pt').input_features
        
        # Generate predicted token ids and decode to text
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
    except Exception as e:
        raise Exception(f'Transcription error: {e}')

# test_function_code --------------------

def test_transcribe_audio():
    print("Testing started.")
    # Assumed example audio file path
    example_audio_file = 'example.wav'

    # Testing case 1: Correct transcription
    print("Testing case [1/2] started.")
    expected_transcription = '...'  # Expected result for the example audio sample
    transcription = transcribe_audio(example_audio_file)
    assert transcription == expected_transcription, f"Test case [1/2] failed: Expected {expected_transcription}, got {transcription}"

    # Testing case 2: FileNotFound exception
    print("Testing case [2/2] started.")
    non_existing_file = 'non_existing.wav'
    try:
        transcribe_audio(non_existing_file)
        assert False, "Test case [2/2] failed: FileNotFoundError was expected."
    except FileNotFoundError:
        # Expected result
        pass
    except Exception as e:
        assert False, f"Test case [2/2] failed: Unexpected exception {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_transcribe_audio()