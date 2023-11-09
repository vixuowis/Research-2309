import torchaudio
from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
import os
import json


def translate_speech_to_speech(audio_file_path):
    """
    This function translates English speech to Hokkien speech using the 'facebook/xm_transformer_s2ut_en-hk' model from Fairseq.
    
    Parameters:
    audio_file_path (str): The path to the English audio file to be translated.
    
    Returns:
    tuple: The translated Hokkien audio and its sample rate.
    """
    cache_dir = os.getenv('HUGGINGFACE_HUB_CACHE')
    models, cfg, task = hub_utils.load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_s2ut_en-hk', arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'}, cache_dir=cache_dir)
    model = models[0].cpu()
    audio, _ = torchaudio.load(audio_file_path)
    sample = S2THubInterface.get_model_input(task, audio)
    unit = S2THubInterface.get_prediction(task, model, generator, sample)
    hkg_vocoder = hub_utils.snapshot_download('facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS', cache_dir=cache_dir)
    x = hub_utils.from_pretrained(hkg_vocoder, 'model.pt', '.', config_yaml='config.json', fp16=False, is_vocoder=True)
    vocoder_cfg = json.load(open(f"{x['args']['data']}/config.json"))
    vocoder = CodeHiFiGANVocoder(x['args']['model_path'][0], vocoder_cfg)
    wav, sr = vocoder(unit)
    return wav, sr