# requirements_file --------------------

import subprocess

requirements = ["fairseq", "huggingface_hub", "torchaudio"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import json
from fairseq import hub_utils
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
import torchaudio
import IPython.display as ipd

# function_code --------------------

def generate_hokkien_speech(text):
    """
    Generates speech audio for the given Hokkien text using a pre-trained TAT-TTS model.

    Args:
        text (str): The Hokkien text to be converted to speech.

    Returns:
        IPython.display.Audio: An IPython Audio object with the generated speech.

    Raises:
        ValueError: If the text input is empty.
    """
    if not text:
        raise ValueError('The text input is empty')

    model_path = hub_utils.from_pretrained(
        'facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS',
        checkpoint_file='model.pt',
        archive_map=CodeHiFiGANVocoder.hub_models(),
        config_yaml='config.json'
    )
    vocoder_cfg = model_path['args']
    vocoder = CodeHiFiGANVocoder(vocoder_cfg.model_path[0], vocoder_cfg)
    tts_model = VocoderHubInterface(vocoder_cfg, vocoder)
    tts_sample = tts_model.get_model_input(text)
    wav, sr = tts_model.get_prediction(tts_sample)
    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_generate_hokkien_speech():
    print("Testing started.")
    text_to_test = '你好'  # Sample Hokkien text

    # Testing case 1: Valid Input
    print("Testing case [1/2] started.")
    audio_result = generate_hokkien_speech(text_to_test)
    assert isinstance(audio_result, ipd.Audio), f"Test case [1/2] failed: Expected an IPython Audio object, got {type(audio_result)}"

    # Testing case 2: Empty Input
    print("Testing case [2/2] started.")
    try:
        generate_hokkien_speech('')
    except ValueError as e:
        assert str(e) == 'The text input is empty', f"Test case [2/2] failed: Expected ValueError with message 'The text input is empty', got {str(e)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_hokkien_speech()