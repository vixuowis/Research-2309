# requirements_file --------------------

import subprocess

requirements = ["fairseq", "torchaudio", "huggingface_hub"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import json
import os
from pathlib import Path
import torchaudio
import IPython.display as ipd
from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download

# function_code --------------------

def translate_hokkien_to_english(audio_path: str) -> ipd.Audio:
    """
    Translates Hokkien speech to English speech using pretrained models.

    Args:
        audio_path (str): The file path to the Hokkien audio file.

    Returns:
        IPython.display.Audio: An IPython display Audio object for the translated audio.

    Raises:
        FileNotFoundError: If the audio file is not found at the provided path.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f'Audio file not found at {audio_path}')

    # Load the model for Hokkien to English translation
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_s2ut_hk-en', arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'})
    model = models[0].cpu()
    cfg['task'].cpu = True
    generator = task.build_generator([model], cfg)

    # Load Hokkien audio
    audio, _ = torchaudio.load(audio_path)

    # Convert to model input
    sample = S2THubInterface.get_model_input(task, audio)
    # Translate speech to English
    translation = S2THubInterface.get_prediction(task, model, generator, sample)

    # Load the vocoder model for text-to-speech
    cache_dir = snapshot_download('facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur')
    x = hub_utils.from_pretrained(cache_dir, 'model.pt', '.', archive_map=CodeHiFiGANVocoder.hub_models(), config_yaml='config.json', fp16=False, is_vocoder=True)
    with open(os.path.join(x['args']['data'], 'config.json')) as f:
        vocoder_cfg = json.load(f)

    vocoder = CodeHiFiGANVocoder(x['args']['model_path'][0], vocoder_cfg)
    tts_model = VocoderHubInterface(vocoder_cfg, vocoder)
    tts_sample = tts_model.get_model_input(translation)

    # Convert translated text back to speech
    wav, sr = tts_model.get_prediction(tts_sample)

    # Return the translated audio
    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_translate_hokkien_to_english():
    print("Testing started.")
    audio_path = 'audio_samples/sample_hokkien.wav'  # FIXME: Replace this with actual test audio path
    sample_data = None # FIXME: Load an actual audio sample or mock up data

    # Mocking a function to load the sample data if necessary
    # sample_data = load_mock_audio_sample(sample_path)

    # Testing case 1: Translate Hokkien audio file to English
    print("Testing case [1/1] started.")
    translated_audio = translate_hokkien_to_english(audio_path)
    # Assuming we have some way to validate the output, replace the condition below
    assert isinstance(translated_audio, ipd.Audio), f"Test case [1/1] failed: The result is not an IPython.display.Audio object"
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_hokkien_to_english()