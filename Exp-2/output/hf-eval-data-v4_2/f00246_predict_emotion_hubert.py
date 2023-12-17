# requirements_file --------------------

!pip install -U audio_models transformers torch numpy pydub mock

# function_import --------------------

from audio_models import HubertForSpeechClassification
from transformers import Wav2Vec2FeatureExtractor, AutoConfig
import torch, torch.nn.functional as F, numpy as np
from pydub import AudioSegment

# function_code --------------------

def predict_emotion_hubert(audio_file):
    """
    Predict the emotion of the speaker in a given audio file.

    Args:
        audio_file (str): The path to the audio file.

    Returns:
        list: A list of dictionaries containing "emo" and "score", representing the predicted emotions and their scores.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    sampling_rate = 16000
    sound = AudioSegment.from_file(audio_file)
    sound = sound.set_frame_rate(sampling_rate)
    sound_array = np.array(sound.get_array_of_samples())
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960')
    inputs = feature_extractor(sound_array, sampling_rate=sampling_rate, return_tensors='pt', padding=True)
    model = HubertForSpeechClassification.from_pretrained('Rajaram1996/Hubert_emotion')
    config = AutoConfig.from_pretrained('Rajaram1996/Hubert_emotion')
    inputs = {key: inputs[key].to('cpu').float() for key in inputs}
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{'emo': config.id2label[i], 'score': round(score * 100, 1)} for i, score in enumerate(scores)]
    return [row for row in sorted(outputs, key=lambda x: x['score'], reverse=True) if row['score'] != 0.0][:2]

# test_function_code --------------------

from mock import MagicMock, patch

@patch('builtins.open', new_callable=MagicMock)
def test_predict_emotion_hubert(mock_open):
    print("Testing started.")
    # Replace the model and feature_extractor with mock for testing
    with patch('{}.HubertForSpeechClassification.from_pretrained'.format('audio_models')) as mock_model,
         patch('{}.Wav2Vec2FeatureExtractor.from_pretrained'.format('transformers')) as mock_extractor:
        audio_file = 'test_audio.wav'
        # Setting up mock for the methods
        model = MagicMock()
        feature_extractor = MagicMock()
        mock_model.return_value = model
        mock_extractor.return_value = feature_extractor
        # Set up fake output for the model
        model.return_value = MagicMock(logits=torch.tensor([[1.0, 2.0]]))
        feature_extractor.return_value = {'input_values': torch.tensor([1.0])}

        expected_output = [{'emo': 'happy', 'score': 66.7}, {'emo': 'sad', 'score': 33.3}]

        real_output = predict_emotion_hubert(audio_file)

        print("Testing finished.")
        assert expected_output == real_output, f"Test failed: Expected {expected_output}, got {real_output}"

# call_test_function_line --------------------

test_predict_emotion_hubert()