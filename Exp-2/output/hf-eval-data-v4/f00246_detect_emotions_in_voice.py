# requirements_file --------------------

!pip install -U audio_models transformers torch numpy pydub

# function_import --------------------

from audio_models import HubertForSpeechClassification
from transformers import Wav2Vec2FeatureExtractor, AutoConfig
import torch, torch.nn.functional as F, numpy as np
from pydub import AudioSegment

# function_code --------------------

def detect_emotions_in_voice(audio_file):
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
    outputs = [{'emo': config.id2label[i], 'score': round(score * 100, 1)} for i, score in enumerate(scores)]
    
    return [row for row in sorted(outputs, key=lambda x: x['score'], reverse=True) if row['score'] != '0.0'][:2]

# test_function_code --------------------

def test_detect_emotions_in_voice():
    print('Testing started.')
    sample_data = 'sample_audio.mp3'  # Replace with an actual audio file path

    print('Testing case [1/1] started.')
    prediction = detect_emotions_in_voice(sample_data)
    assert prediction and isinstance(prediction, list), 'Test case [1/1] failed: the function should return a list of predictions.'
    print('Prediction:', prediction)
    print('Testing finished.')

test_detect_emotions_in_voice()