from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch


def speech_emotion_recognition(audio_files):
    """
    Analyze the emotional speech of public speaking practice sessions.

    Args:
        audio_files (list): List of paths to the recorded audio files.

    Returns:
        ser_outputs (list): List of detected emotions in the speech.
    """
    model = Wav2Vec2ForCTC.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    ser_outputs = []
    for audio in audio_files:
        processor = Wav2Vec2Processor.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
        input_features = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(**input_features).logits
        predicted_emotion = torch.argmax(logits, dim=-1).item()
        ser_outputs.append(emotions[predicted_emotion])
    return ser_outputs