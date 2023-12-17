# requirements_file --------------------

import subprocess

requirements = ["fairseq", "huggingface_hub", "torchaudio"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import os
import torchaudio
from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text import S2THubInterface

# function_code --------------------

def translate_english_to_hokkien(audio_file_path: str) -> str:
    """
    Translates an English audio file to Hokkien using the Facebook XM Transformer model.

    Args:
        audio_file_path: The file path to the English audio file to be translated.

    Returns:
        A string representing the file path of the translated Hokkien audio.

    Raises:
        FileNotFoundError: If the given audio file path does not exist.
    """
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f'The audio file {audio_file_path} does not exist.')

    cache_dir = os.getenv('HUGGINGFACE_HUB_CACHE')
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_s2ut_en-hk', arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'}, cache_dir=cache_dir)
    model = models[0].cpu()
    cfg['task'].cpu = True
    generator = task.build_generator([model], cfg)

    english_audio, _ = torchaudio.load(audio_file_path)
    english_sample = S2THubInterface.get_model_input(task, english_audio)
    hokkien_translation = S2THubInterface.get_prediction(task, model, generator, english_sample)
    
    output_file_path = 'translated_hokkien.wav'
    torchaudio.save(output_file_path, hokkien_translation, sample_rate=48000)
    return output_file_path

# test_function_code --------------------

def test_translate_english_to_hokkien():
    print("Testing started.")
    sample_audio_file = 'sample_english_audio.wav'  # This is an example and should be replaced with actual file path

    print("Testing case [1/1] started.")
    translated_file_path = translate_english_to_hokkien(sample_audio_file)
    assert os.path.isfile(translated_file_path), f"Test case [1/1] failed: Translated file {translated_file_path} does not exist."
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_english_to_hokkien()