# function_import --------------------

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# function_code --------------------

def speech_emotion_recognition(audio_files):
    """
    Analyze the emotional speech in the given audio files.

    Args:
        audio_files (list): A list of paths to the audio files to be analyzed.

    Returns:
        list: A list of detected emotions in the speech of each audio file.

    Raises:
        OSError: If there is a problem with the file path or reading the file.
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
    Test the speech_emotion_recognition function with some example audio files.
    """
    sample_audio_files = ['sample1.wav', 'sample2.wav', 'sample3.wav']
    emotions = speech_emotion_recognition(sample_audio_files)
    assert isinstance(emotions, list), 'The output should be a list.'
    assert len(emotions) == len(sample_audio_files), 'The output list length should match the input list length.'
    assert all(emotion in ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'] for emotion in emotions), 'All emotions should be in the predefined list.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_speech_emotion_recognition()