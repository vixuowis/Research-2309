import torchaudio
import IPython.display as ipd
from fairseq import hub_utils
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface


def hokkien_text_to_speech(text_input):
    """
    This function converts Hokkien text to speech using the pre-trained model 'facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS'.
    
    Parameters:
    text_input (str): The Hokkien text to be converted to speech.
    
    Returns:
    Audio: The audio output of the converted text.
    """
    model_path = load_model_ensemble_and_task_from_hf_hub('facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS')
    vocoder_cfg = model_path['args']
    vocoder = CodeHiFiGANVocoder(model_path['args']['model_path'][0], vocoder_cfg)
    tts_model = VocoderHubInterface(vocoder_cfg, vocoder)
    tts_sample = tts_model.get_model_input(text_input)
    wav, sr = tts_model.get_prediction(tts_sample)
    return ipd.Audio(wav, rate=sr)