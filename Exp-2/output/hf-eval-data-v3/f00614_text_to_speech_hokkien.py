# function_import --------------------

from fairseq import hub_utils
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
import torchaudio
import IPython.display as ipd

# function_code --------------------

def text_to_speech_hokkien(text_input):
    '''
    Converts Hokkien text to speech using the TAT-TTS dataset.

    Args:
        text_input (str): The Hokkien text to be converted to speech.

    Returns:
        IPython.lib.display.Audio: The audio output of the converted text.
    '''
    model_path = hub_utils.load_model_ensemble_and_task_from_hf_hub('facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS')
    vocoder_cfg = model_path['args']
    vocoder = CodeHiFiGANVocoder(model_path['args']['model_path'][0], vocoder_cfg)

    tts_model = VocoderHubInterface(vocoder_cfg, vocoder)
    tts_sample = tts_model.get_model_input(text_input)

    wav, sr = tts_model.get_prediction(tts_sample)
    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_text_to_speech_hokkien():
    '''
    Tests the text_to_speech_hokkien function.
    '''
    test_text = 'Insert Hokkien text here'
    audio_output = text_to_speech_hokkien(test_text)
    assert isinstance(audio_output, ipd.lib.display.Audio), 'Output is not an audio file.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_text_to_speech_hokkien()