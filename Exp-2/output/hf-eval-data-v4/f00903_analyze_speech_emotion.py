# requirements_file --------------------

!pip install -U transformers==4.8.2 torch

# function_import --------------------

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# function_code --------------------

def analyze_speech_emotion(audio_file_path, model, processor):
    """
    Analyze the emotion of a speech sample from an audio file.

    Parameters:
    - audio_file_path (str): Path to the audio file.
    - model (Wav2Vec2ForCTC): Pretrained speech emotion recognition model.
    - processor (Wav2Vec2Processor): Processor to convert audio to model input format.

    Returns:
    - str: The detected emotion label.
    """
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    input_features = processor(audio_file_path, return_tensors="pt", padding=True, sampling_rate=16000)
    logits = model(**input_features).logits
    predicted_emotion_index = torch.argmax(logits, dim=-1).item()
    return emotions[predicted_emotion_index]

# test_function_code --------------------

def test_analyze_speech_emotion():
    print("Testing analyze_speech_emotion function.")
    model = Wav2Vec2ForCTC.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
    processor = Wav2Vec2Processor.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')

    # Test case with an example audio file
    audio_file_path = 'example.wav'
    detected_emotion = analyze_speech_emotion(audio_file_path, model, processor)

    assert isinstance(detected_emotion, str), f"Expected string output, got {type(detected_emotion)}"
    print("Test passed successfully!")

test_analyze_speech_emotion()