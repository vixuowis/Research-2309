# function_import --------------------

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# function_code --------------------

def analyze_emotion(audio_path):
    """
    Analyze the emotion of the speaker in an audio recording.

    Args:
        audio_path (str): The path to the audio file to analyze.

    Returns:
        str: The predicted emotion of the speaker.

    Raises:
        OSError: If there is not enough space on the device to download the model.
    """
    model = Wav2Vec2ForCTC.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
    tokenizer = Wav2Vec2Processor.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')

    input_data = tokenizer(audio_path, return_tensors='pt')
    input_values = input_data.input_values.to('cuda')
    predictions = model(input_values)
    predicted_ids = torch.argmax(predictions.logits, dim=-1)
    predicted_emotions = tokenizer.batch_decode(predicted_ids)

    return predicted_emotions[0]

# test_function_code --------------------

def test_analyze_emotion():
    """Test the analyze_emotion function."""
    test_audio_path = 'path/to/test/audiofile.wav'
    emotion = analyze_emotion(test_audio_path)
    assert isinstance(emotion, str), 'The function should return a string.'
    assert emotion in ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'], 'The function should return a valid emotion.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_analyze_emotion()