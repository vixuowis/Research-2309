from f00456_get_audio_path import *
def test_get_audio_path():
    data = {
        'audio': {
            'array': array([-0.00024414,  0.        ,  0.        , ...,  0.00024414,
                  0.00024414,  0.00024414], dtype=float32),
            'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
            'sampling_rate': 8000
        },
        'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
        'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"
    }
    assert get_audio_path(data) == '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav'

test_get_audio_path()
