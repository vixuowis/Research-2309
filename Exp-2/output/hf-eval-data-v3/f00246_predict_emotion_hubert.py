# function_import --------------------

from transformers import Wav2Vec2FeatureExtractor, AutoConfig
import torch, torch.nn.functional as F, numpy as np
from pydub import AudioSegment

# function_code --------------------

def predict_emotion_hubert(audio_file):
    """
    This function predicts the emotion in a given audio file using the pre-trained Hubert model.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        list: A list of dictionaries containing the predicted emotions and their corresponding scores.
    """
    from transformers import HubertForSpeechClassification
    model = HubertForSpeechClassification.from_pretrained('Rajaram1996/Hubert_emotion')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960')
    sampling_rate = 16000
    config = AutoConfig.from_pretrained('Rajaram1996/Hubert_emotion')

    def speech_file_to_array(path, sampling_rate):
        sound = AudioSegment.from_file(path)
        sound = sound.set_frame_rate(sampling_rate)
        sound_array = np.array(sound.get_array_of_samples())
        return sound_array

    sound_array = speech_file_to_array(audio_file, sampling_rate)
    inputs = feature_extractor(sound_array, sampling_rate=sampling_rate, return_tensors='pt', padding=True)
    inputs = {key: inputs[key].to('cpu').float() for key in inputs}
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"emo": config.id2label[i], "score": round(score * 100, 1)} for i, score in enumerate(scores)]
    return [row for row in sorted(outputs, key=lambda x: x['score'], reverse=True) if row['score'] != '0.0'][:2]

# test_function_code --------------------

def test_predict_emotion_hubert():
    """
    This function tests the predict_emotion_hubert function.
    """
    result = predict_emotion_hubert('test_audio.mp3')
    assert isinstance(result, list), 'The result should be a list.'
    assert len(result) > 0, 'The result list should not be empty.'
    for item in result:
        assert isinstance(item, dict), 'Each item in the result list should be a dictionary.'
        assert 'emo' in item, 'Each item should have an "emo" key.'
        assert 'score' in item, 'Each item should have a "score" key.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_emotion_hubert()