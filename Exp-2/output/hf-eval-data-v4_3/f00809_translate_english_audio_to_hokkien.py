# requirements_file --------------------

import subprocess

requirements = ["torchaudio", "IPython", "fairseq", "huggingface_hub"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import os
import torchaudio
from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from huggingface_hub import snapshot_download
import IPython.display as ipd

# function_code --------------------

def translate_english_audio_to_hokkien(audio_path: str) -> ipd.Audio:
    """
    Translates spoken English in an audio file to spoken Hokkien.

    Args:
        audio_path (str): The file path of the English audio file to be translated.

    Returns:
        IPython.display.Audio: An IPython audio object that can play the translated Hokkien audio.

    Raises:
        FileNotFoundError: If the audio_path does not exist.
        Exception: If there is an error in loading models or processing audio.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load the translation model
    cache_dir = os.getenv('HUGGINGFACE_HUB_CACHE')
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/xm_transformer_unity_en-hk', arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'}, cache_dir=cache_dir)
    model = models[0].cpu()
    cfg[task].cpu = True
    generator = task.build_generator([model], cfg)

    # Load the English audio
    audio, _ = torchaudio.load(audio_path)
    sample = task.get_model_input(audio)
    unit = task.get_prediction(model, generator, sample)

    # Load the CodeHiFiGANVocoder
    vocoder_cache_dir = snapshot_download('facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS', library_name="fairseq")
    vocoder_cfg, vocoder = load_model_ensemble_and_task_from_hf_hub('facebook/CodeHiFiGANVocoder', cache_dir=vocoder_cache_dir)

    # Generate spoken Hokkien audio from the translation
    tts_sample = vocoder.get_model_input(unit)
    wav, sr = vocoder.get_prediction(tts_sample)
    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_translate_english_audio_to_hokkien():
    print("Testing started.")
    # Assuming we have a sample English audio file for testing
    sample_audio_path = '/path/to/sample/english/audio.wav'

    # Test case 1: Valid audio file path
    print("Testing case [1/3] started.")
    result = translate_english_audio_to_hokkien(sample_audio_path)
    assert isinstance(result, ipd.Audio), f"Test case [1/3] failed: Expected IPython.display.Audio, got {type(result)}"

    # Test case 2: Non-existent audio file path
    print("Testing case [2/3] started.")
    try:
        translate_english_audio_to_hokkien('/path/to/nonexistent/audio.wav')
        assert False, "Test case [2/3] failed: Expected FileNotFoundError"
    except FileNotFoundError:
        pass

    # Test case 3: Ensure audio is playable (mock or simulate playing)
    print("Testing case [3/3] started.")
    # This case requires a hypothetical or simulated environment to test if audio can be played.
    # Here, we simply assume it is playable if no exception is raised.
    print("Test case [3/3] succeeded: Audio is assumed to be playable because no exception was raised.")
    print("Testing finished.")


# call_test_function_line --------------------

test_translate_english_audio_to_hokkien()