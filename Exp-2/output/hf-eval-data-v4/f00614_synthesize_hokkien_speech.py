# requirements_file --------------------

!pip install -U fairseq torchaudio huggingface_hub IPython

# function_import --------------------

from fairseq import hub_utils
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
import torchaudio
import IPython.display as ipd

# function_code --------------------

def synthesize_hokkien_speech(text_input):
    """
    Synthesize Hokkien speech from text using a Fairseq TTS model.

    :param text_input: String containing the text in Hokkien dialect to be synthesized
    :return: IPython.display.Audio object containing the synthesized speech audio
    """
    model_path = hub_utils.from_pretrained(
        'facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS',
         archive_map=CodeHiFiGANVocoder.hub_models(),
         config_yaml='config.json'
    )
    vocoder_cfg = model_path['args']
    vocoder = CodeHiFiGANVocoder(model_path['args']['model_path'][0], vocoder_cfg)

    tts_model = VocoderHubInterface(vocoder_cfg, vocoder)
    tts_sample = tts_model.get_model_input(text_input)
    wav, sr = tts_model.get_prediction(tts_sample)
    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_synthesize_hokkien_speech():
    print("Testing Hokkien speech synthesis.")

    # Test with a known Hokkien phrase
    hokkien_phrase = ""
    audio_result = synthesize_hokkien_speech(hokkien_phrase)

    # No easy way to validate the audio output in an automated test, so here we just check the type
    print("Test started.")
    assert isinstance(audio_result, ipd.Audio), f"Test failed: The output should be an IPython.display.Audio object"
    print("Test passed.")

test_synthesize_hokkien_speech()