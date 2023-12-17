# requirements_file --------------------

import subprocess

requirements = ["fairseq", "huggingface_hub", "torchaudio"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import json
import os
import torchaudio
from pathlib import Path
from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download

# function_code --------------------

def translate_english_to_hokkien(audio_path: str) -> dict:
    """
    Translates English speech input to Hokkien speech output in real-time.

    Args:
        audio_path (str): The file path to the English speech audio file.

    Returns:
        dict: A dictionary containing 'wav': synthesized Hokkien speech audio, 'sr': sample rate.

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        Exception: If translation or speech synthesis fails.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cache_dir = os.getenv('HUGGINGFACE_HUB_CACHE')
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_s2ut_en-hk', arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'}, cache_dir=cache_dir)
    model = models[0].cpu()
    cfg['task'].cpu = True
    generator = task.build_generator([model], cfg)

    audio, _ = torchaudio.load(audio_path)
    sample = S2THubInterface.get_model_input(task, audio)
    unit = S2THubInterface.get_prediction(task, model, generator, sample)

    library_name = 'fairseq'
    cache_dir = (cache_dir or (Path.home() / '.cache' / library_name).as_posix())
    cache_dir = snapshot_download('facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS', cache_dir=cache_dir, library_name=library_name)
    x = hub_utils.from_pretrained(cache_dir, 'model.pt', '.', archive_map=CodeHiFiGANVocoder.hub_models(), config_yaml='config.json', fp16=False, is_vocoder=True)
    with open(f'{x['args']['data']}/config.json') as f:
      vocoder_cfg = json.load(f)
    assert (len(x['args']['model_path']) == 1), 'Too many vocoder models in the input'
    vocoder = CodeHiFiGANVocoder(x['args']['model_path'][0], vocoder_cfg)
    tts_model = VocoderHubInterface(vocoder_cfg, vocoder)
    tts_sample = tts_model.get_model_input(unit)
    wav, sr = tts_model.get_prediction(tts_sample)
    return {'wav': wav, 'sr': sr}

# test_function_code --------------------

def test_translate_english_to_hokkien():
    print("Testing started.")
    sample_audio_path = 'test_english_audio.wav'  # Sample audio path for testing

    # Test case 1: File not found
    print("Testing case [1/3] started.")
    try:
        translate_english_to_hokkien('non_existent_file.wav')
        assert False, "Test case [1/3] failed: FileNotFoundError not raised for non-existent file"
    except FileNotFoundError:
        assert True

    # Test case 2: Translation succeeds
    print("Testing case [2/3] started.")
    try:
        result = translate_english_to_hokkien(sample_audio_path)
        assert 'wav' in result and 'sr' in result, f"Test case [2/3] failed: Missing 'wav' or 'sr' in the result"
    except Exception as e:
        assert False, f"Test case [2/3] failed: Exception raised - {e}"

    # Test case 3: Speech synthesis succeeds
    print("Testing case [3/3] started.")
    try:
        result = translate_english_to_hokkien(sample_audio_path)
        assert isinstance(result['wav'], np.ndarray), "Test case [3/3] failed: 'wav' is not a NumPy array"
        assert isinstance(result['sr'], int), "Test case [3/3] failed: 'sr' is not an integer"
    except Exception as e:
        assert False, f"Test case [3/3] failed: Exception raised - {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_english_to_hokkien()