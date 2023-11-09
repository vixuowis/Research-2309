# function_import --------------------

from fairseq import hub_utils
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
import torchaudio
import IPython.display as ipd

# function_code --------------------

def text_to_speech_hokkien(text_input):
    """
    This function converts Hokkien text to speech using a pre-trained model from Fairseq.

    Args:
        text_input (str): The Hokkien text to be converted to speech.

    Returns:
        IPython.lib.display.Audio: An audio object that can be played in a Jupyter notebook.
    """
    model_path = load_model_ensemble_and_task_from_hf_hub('facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS')
    vocoder_cfg = model_path['args']
    vocoder = CodeHiFiGANVocoder(model_path['args']['model_path'][0], vocoder_cfg)

    tts_model = VocoderHubInterface(vocoder_cfg, vocoder)
    tts_sample = tts_model.get_model_input(text_input)

    wav, sr = tts_model.get_prediction(tts_sample)
    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_text_to_speech_hokkien():
    """
    This function tests the text_to_speech_hokkien function by passing a sample Hokkien text and checking the type of the output.
    """
    sample_text = 'Insert Hokkien text here'
    output = text_to_speech_hokkien(sample_text)
    assert isinstance(output, ipd.lib.display.Audio), 'Output is not an audio object'

# call_test_function_code --------------------

test_text_to_speech_hokkien()