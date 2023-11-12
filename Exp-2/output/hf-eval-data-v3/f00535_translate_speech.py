# function_import --------------------

import json
import os
import torchaudio
import IPython.display as ipd
from pathlib import Path
from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download

# function_code --------------------

def translate_speech(audio_file_path):
    """
    Translates spoken language from Hokkien to English.

    Args:
        audio_file_path (str): The path to the audio file in Hokkien.

    Returns:
        IPython.lib.display.Audio: The translated audio in English.
    """
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_s2ut_hk-en', arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'})
    model = models[0].cpu()
    cfg['task'].cpu = True
    generator = task.build_generator([model], cfg)
    audio, _ = torchaudio.load(audio_file_path)

    sample = S2THubInterface.get_model_input(task, audio)
    translation = S2THubInterface.get_prediction(task, model, generator, sample)
    cache_dir = snapshot_download('facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur')

    x = hub_utils.from_pretrained(cache_dir, 'model.pt', '.', archive_map=CodeHiFiGANVocoder.hub_models(), config_yaml='config.json', fp16=False, is_vocoder=True)
    with open(os.path.join(x['args']['data'], 'config.json')) as f:
        vocoder_cfg = json.load(f)

    vocoder = CodeHiFiGANVocoder(x['args']['model_path'][0], vocoder_cfg)
    tts_model = VocoderHubInterface(vocoder_cfg, vocoder)
    tts_sample = tts_model.get_model_input(translation)
    wav, sr = tts_model.get_prediction(tts_sample)
    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_translate_speech():
    """
    Tests the translate_speech function.
    """
    # Test with a sample audio file
    audio_file_path = '/path/to/sample/audio/file.wav'
    result = translate_speech(audio_file_path)
    assert isinstance(result, ipd.Audio), 'The result should be an instance of IPython.lib.display.Audio.'

    # Test with a non-existing audio file
    audio_file_path = '/path/to/non/existing/audio/file.wav'
    try:
        result = translate_speech(audio_file_path)
    except Exception as e:
        assert isinstance(e, FileNotFoundError), 'A FileNotFoundError should be raised if the audio file does not exist.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_speech()