# function_import --------------------

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# function_code --------------------

def speech_emotion_recognition(audio_files):
    """
    This function uses a pre-trained model from Hugging Face Transformers to perform speech emotion recognition.

    Args:
        audio_files (list): A list of paths to the audio files to be analyzed.

    Returns:
        ser_outputs (list): A list of predicted emotions for each audio file.
    """
    model = Wav2Vec2ForCTC.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    ser_outputs = []
    for audio in audio_files:
        processor = Wav2Vec2Processor.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
        input_features = processor(audio, return_tensors='pt', padding=True, sampling_rate=16000)
        logits = model(**input_features).logits
        predicted_emotion = torch.argmax(logits, dim=-1).item()
        ser_outputs.append(emotions[predicted_emotion])
    return ser_outputs

# test_function_code --------------------

def test_speech_emotion_recognition():
    """
    This function tests the speech_emotion_recognition function by using a sample audio file.
    """
    sample_audio_files = ['sample1.wav', 'sample2.wav']
    emotions = speech_emotion_recognition(sample_audio_files)
    assert isinstance(emotions, list), 'The output should be a list.'
    assert all(isinstance(emotion, str) for emotion in emotions), 'All elements in the output list should be strings.'
    assert all(emotion in ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'] for emotion in emotions), 'All elements in the output list should be valid emotions.'

# call_test_function_code --------------------

test_speech_emotion_recognition()