# function_import --------------------

import os
from pathlib import Path
import torchaudio
from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download
import IPython.display as ipd

# function_code --------------------

def speech_to_speech_translation(audio_file_path):
    """
    Perform speech-to-speech translation between Hokkien and English using the xm_transformer_s2ut_hk-en model.

    Args:
        audio_file_path (str): The path to the input audio file.

    Returns:
        IPython.lib.display.Audio: The translated audio.
    """
    cache_dir = os.getenv('HUGGINGFACE_HUB_CACHE')
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/xm_transformer_s2ut_hk-en',
        arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'},
        cache_dir=cache_dir
    )
    model = models[0].cpu()
    generator = task.build_generator([model], cfg)
    audio, _ = torchaudio.load(audio_file_path)
    sample = S2THubInterface.get_model_input(task, audio)
    unit = S2THubInterface.get_prediction(task, model, generator, sample)

    cache_dir = snapshot_download('facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur', cache_dir=cache_dir)

    x = hub_utils.from_pretrained(
        cache_dir,
        'model.pt',
        '.',
        archive_map=CodeHiFiGANVocoder.hub_models(),
        config_yaml='config.json'
    )
    vocoder = CodeHiFiGANVocoder(x['args']['model_path'][0], x['config'])
    tts_model = VocoderHubInterface(x['config'], vocoder)
    tts_sample = tts_model.get_model_input(unit)
    wav, sr = tts_model.get_prediction(tts_sample)

    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_speech_to_speech_translation():
    """
    Test the speech_to_speech_translation function.
    """
    # Test with a sample audio file
    audio_file_path = '/path/to/an/audio/file'
    result = speech_to_speech_translation(audio_file_path)
    assert isinstance(result, ipd.Audio), 'The result should be an instance of IPython.lib.display.Audio.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_speech_to_speech_translation()