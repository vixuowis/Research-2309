# requirements_file --------------------

!pip install -U transformers datasets torch librosa

# function_import --------------------

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# function_code --------------------

def transcribe_audio(audio_path):
    """
    Transcribes the content of an audio file to text using a pre-trained Wav2Vec2 model.

    Args:
        audio_path (str): The path to the audio file to be transcribed.

    Returns:
        str: The transcription of the audio file.

    Raises:
        FileNotFoundError: If the audio file does not exist at the provided path.
        RuntimeError: If the audio processing or transcription fails.
    """
    # Load pre-trained Wav2Vec2 model and processor
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

    # Check if the file exists
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f'The audio file was not found: {audio_path}')

    # Load the audio file using librosa
    input_values = processor(audio_path, return_tensors='pt', padding='longest').input_values

    # Get logits from the model
    logits = model(input_values).logits

    # Predict the transcriptions
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode transcriptions into text
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    print("Testing started.")
    # Load dataset with audio files
    dataset = load_dataset('patrickvonplaten/librispeech_asr_dummy', 'clean', split='validation')
    sample_data = dataset[0]

    # Test case 1: Transcribe a valid audio file
    print("Testing case [1/1] started.")
    try:
        transcription = transcribe_audio(sample_data['file'])
        assert isinstance(transcription, str), 'The transcription should be a string.'
    except Exception as e:
        print(f"Test case [1/1] failed: {e}")
    else:
        print("Test case [1/1] passed.")
    print("Testing finished.")

# call_test_function_line --------------------

test_transcribe_audio()