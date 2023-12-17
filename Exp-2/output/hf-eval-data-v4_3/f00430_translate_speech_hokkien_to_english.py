# requirements_file --------------------

import subprocess

requirements = ["fairseq", "torchaudio", "huggingface_hub", "IPython"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import os
import IPython.display as ipd
import torchaudio
from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download

# function_code --------------------

def translate_speech_hokkien_to_english(audio_path: str) -> ipd.Audio:
    """
    Perform speech-to-speech translation from Hokkien to English.

    Args:
        audio_path (str): The path to the input audio file in Hokkien.

    Returns:
        IPython.display.Audio: An object that contains the translated English audio data.

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        RuntimeError: If the translation process fails.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}.")

    cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE")
    # Load the speech-to-text model
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/xm_transformer_s2ut_hk-en',
        arg_overrides={"config_yaml": "config.yaml", "task": "speech_to_text"},
        cache_dir=cache_dir
    )
    model = models[0].cpu()
    generator = task.build_generator([model], cfg)

    # Load input audio
    audio, _ = torchaudio.load(audio_path)
    sample = S2THubInterface.get_model_input(task, audio)
    unit = S2THubInterface.get_prediction(task, model, generator, sample)

    # Load vocoder for text-to-speech
    cache_dir = snapshot_download("facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur", cache_dir=cache_dir)

    # Load the vocoder model
    x = hub_utils.from_pretrained(
        cache_dir,
        "model.pt",
        ".",
        archive_map=CodeHiFiGANVocoder.hub_models(),
        config_yaml="config.json"
    )
    vocoder = CodeHiFiGANVocoder(x["args"]["model_path"][0], x["config"])
    tts_model = VocoderHubInterface(x["config"], vocoder)
    tts_sample = tts_model.get_model_input(unit)
    wav, sr = tts_model.get_prediction(tts_sample)
    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

import os
def test_translate_speech_hokkien_to_english():
    print("Testing started.")
    # Test with an existing audio file
    audio_path = 'test_hokkien_audio.wav'
    if not os.path.exists(audio_path):
        raise ValueError('Test audio file does not exist.')

    # Testing case 1
    print("Testing case [1/1] started.")
    try:
        result = translate_speech_hokkien_to_english(audio_path)
        assert isinstance(result, ipd.Audio), f"Test case [1/1] failed: The result is not a valid IPython Audio object"
    except Exception as e:
        assert False, f"Test case [1/1] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_speech_hokkien_to_english()