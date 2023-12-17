# requirements_file --------------------

!pip install -U fairseq torchaudio IPython huggingface_hub

# function_import --------------------

import os
import torchaudio
import IPython.display as ipd
from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from huggingface_hub import snapshot_download


# function_code --------------------

def translate_english_to_hokkien(audio_file_path):
    """
    Translates spoken English to spoken Hokkien for an audio file.

    Parameters:
        audio_file_path (str): The file path to the input English audio file.

    Returns:
        IPython.display.Audio: An audio object that can be played in an IPython notebook.
    """
    cache_dir = os.getenv('HUGGINGFACE_HUB_CACHE')
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_unity_en-hk', arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'}, cache_dir=cache_dir)

    audio, _ = torchaudio.load(audio_file_path)
    sample = task.build_generator([models[0]], cfg).get_model_input(audio)
    hokkien_translation = task.get_prediction(models[0], sample)

    vocoder_cache_dir = snapshot_download('facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS', library_name='fairseq')
    vocoder_cfg, vocoder = load_model_ensemble_and_task_from_hf_hub('facebook/CodeHiFiGANVocoder', cache_dir=vocoder_cache_dir)

    tts_sample = vocoder.get_model_input(hokkien_translation)
    wav, sr = vocoder.get_prediction(tts_sample)

    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_translate_english_to_hokkien():
    print('Testing function translate_english_to_hokkien.')
    test_audio_path = '/path/to/an/english/audio/file'

    print('Running translation...')
    translated_audio = translate_english_to_hokkien(test_audio_path)

    print('Checking the result...')
    assert isinstance(translated_audio, ipd.Audio), 'The result should be an IPython.display.Audio object.'
    assert translated_audio.data is not None, 'The audio data should not be None.'
    print('Test successful!')

# Execute the test
test_translate_english_to_hokkien()