# requirements_file --------------------

!pip install -U fairseq torchaudio huggingface_hub

# function_import --------------------

import os
from fairseq import hub_utils
import torchaudio
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download
import IPython.display as ipd

# function_code --------------------

def translate_speech_hokkien_english(audio_file_path):
    """
    Translate speech from Hokkien to English using speech-to-speech model.

    Args:
    audio_file_path (str): The file path to the Hokkien audio file to be translated.

    Returns:
    IPython.display.Audio: The IPython Audio object containing the translated English speech.
    """
    cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE")
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/xm_transformer_s2ut_hk-en',
        arg_overrides={"config_yaml": "config.yaml", "task": "speech_to_text"},
        cache_dir=cache_dir
    )
    model = models[0].cpu()
    generator = task.build_generator([model], cfg)
    audio, _ = torchaudio.load(audio_file_path)
    sample = S2THubInterface.get_model_input(task, audio)
    hokkien_transcript = S2THubInterface.get_prediction(task, model, generator, sample)

    cache_dir = snapshot_download(
        'facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur',
        cache_dir=cache_dir
    )
    x = hub_utils.from_pretrained(
        cache_dir,
        'model.pt',
        '.',
        archive_map=CodeHiFiGANVocoder.hub_models(),
        config_yaml='config.json'
    )
    vocoder = CodeHiFiGANVocoder(x['args']['model_path'][0], x['config'])
    tts_model = VocoderHubInterface(x['config'], vocoder)
    tts_sample = tts_model.get_model_input(hokkien_transcript)
    wav, sr = tts_model.get_prediction(tts_sample)
    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_translate_speech_hokkien_english():
    print("Testing started.")
    output = translate_speech_hokkien_english("/path/to/sample/hokkien/audio.wav")

    # 测试用例 1：检查输出类型
    print("Testing case [1/1] started.")
    assert isinstance(output, ipd.Audio), f"Test case [1/1] failed: The output is not of type IPython.display.Audio"
    print("Testing finished.")

# 运行测试函数
test_translate_speech_hokkien_english()