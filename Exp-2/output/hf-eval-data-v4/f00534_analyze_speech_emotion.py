# requirements_file --------------------

!pip install -U librosa, torch, transformers

# function_import --------------------

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

# function_code --------------------

def analyze_speech_emotion(file_path: str) -> str:
    """
    Analyze the emotion in the speech contained in the provided audio file.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        str: The predicted emotion.
    """
    # Load the pre-trained model and tokenizer
    model = Wav2Vec2ForCTC.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
    tokenizer = Wav2Vec2Processor.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')

    # Load and preprocess the audio file
    audio, rate = librosa.load(file_path, sr=16000)
    input_values = tokenizer(audio, return_tensors='pt', padding='longest').input_values

    # Get predictions
    with torch.no_grad():
        logits = model(input_values).logits

    # Interpret the predictions
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_emotions = tokenizer.batch_decode(predicted_ids)[0]

    return predicted_emotions

# test_function_code --------------------

def test_analyze_speech_emotion():
    print("Testing started.")
    test_audio_path = "test_audio.wav"  # Replace with path to an actual audio file for testing

    # Test case 1: Check the return type
    print("Testing case [1/1] started.")
    result = analyze_speech_emotion(test_audio_path)
    assert isinstance(result, str), f"Test case [1/1] failed: Expected result to be of type str, got {type(result)}"
    print("Testing finished.")