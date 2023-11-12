# function_import --------------------

import os
import json
import torchaudio
from fairseq import hub_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder

# function_code --------------------

def translate_speech_to_speech(audio_file_path: str, model_name: str = 'facebook/xm_transformer_s2ut_en-hk', cache_dir: str = None) -> tuple:
    """
    Translates English speech input to Hokkien in real-time using a pre-trained model.

    Args:
        audio_file_path (str): Path to the English audio file to be translated.
        model_name (str, optional): Name of the pre-trained model to use for translation. Defaults to 'facebook/xm_transformer_s2ut_en-hk'.
        cache_dir (str, optional): Directory to cache the pre-trained model. If None, uses the HUGGINGFACE_HUB_CACHE environment variable. Defaults to None.

    Returns:
        tuple: A tuple containing the translated Hokkien speech units and the sample rate.
    """
    if cache_dir is None:
        cache_dir = os.getenv('HUGGINGFACE_HUB_CACHE')
    models, cfg, task = hub_utils.load_model_ensemble_and_task_from_hf_hub(model_name, arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'}, cache_dir=cache_dir)
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

# test_function_code --------------------

def test_translate_speech_to_speech():
    """Tests the translate_speech_to_speech function."""
    # Test with a sample English audio file
    wav, sr = translate_speech_to_speech('/path/to/sample/english/audio/file')
    assert isinstance(wav, np.ndarray)
    assert isinstance(sr, int)
    # Test with a different model
    wav, sr = translate_speech_to_speech('/path/to/sample/english/audio/file', model_name='facebook/another_model')
    assert isinstance(wav, np.ndarray)
    assert isinstance(sr, int)
    # Test with a specified cache directory
    wav, sr = translate_speech_to_speech('/path/to/sample/english/audio/file', cache_dir='/path/to/cache/directory')
    assert isinstance(wav, np.ndarray)
    assert isinstance(sr, int)
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_translate_speech_to_speech())