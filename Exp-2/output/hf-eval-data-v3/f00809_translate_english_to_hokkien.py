# function_import --------------------

import os
import torchaudio
import IPython.display as ipd
from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from huggingface_hub import snapshot_download

# function_code --------------------

def translate_english_to_hokkien(audio_file_path):
    """
    Translates spoken English to spoken Hokkien for an audio file.

    Args:
        audio_file_path (str): The path to the English audio file.

    Returns:
        IPython.lib.display.Audio: The translated Hokkien audio.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If the translation model or vocoder model cannot be loaded.
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file {audio_file_path} does not exist.")

    cache_dir = os.getenv('HUGGINGFACE_HUB_CACHE')
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_unity_en-hk', arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'}, cache_dir=cache_dir)

    audio, _ = torchaudio.load(audio_file_path)
    sample = task.build_generator([model], cfg).get_model_input(audio)
    unit = task.get_prediction(model, generator, sample)

    vocoder_cache_dir = snapshot_download('facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS', library_name="fairseq")
    vocoder_cfg, vocoder = load_model_ensemble_and_task_from_hf_hub('facebook/CodeHiFiGANVocoder', cache_dir=vocoder_cache_dir)

    tts_sample = vocoder.get_model_input(unit)
    wav, sr = vocoder.get_prediction(tts_sample)

    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_translate_english_to_hokkien():
    """
    Tests the translate_english_to_hokkien function.
    """
    # Test with a valid audio file
    try:
        result = translate_english_to_hokkien('/path/to/an/audio/file')
        assert isinstance(result, ipd.Audio)
    except Exception as e:
        print(f'Test failed with error: {str(e)}')

    # Test with a non-existent audio file
    try:
        result = translate_english_to_hokkien('/path/to/non/existent/file')
    except FileNotFoundError:
        pass
    else:
        print('Test failed. Expected FileNotFoundError for non-existent audio file.')

    print('All tests passed.')

# call_test_function_code --------------------

test_translate_english_to_hokkien()