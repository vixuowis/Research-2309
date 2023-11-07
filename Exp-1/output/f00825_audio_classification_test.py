from f00825_audio_classification import *
def test_audio_classification():
    audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
    expected_preds = [{'score': 0.4532, 'label': 'hap'},
                     {'score': 0.3622, 'label': 'sad'},
                     {'score': 0.0943, 'label': 'neu'},
                     {'score': 0.0903, 'label': 'ang'}]
    assert audio_classification(audio_url) == expected_preds

test_audio_classification()
