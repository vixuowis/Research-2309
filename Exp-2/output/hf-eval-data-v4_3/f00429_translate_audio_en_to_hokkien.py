# requirements_file --------------------

import subprocess

requirements = ["fairseq", "huggingface_hub", "torchaudio"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import os
import torchaudio
from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download

# function_code --------------------

def translate_audio_en_to_hokkien(audio_path, output_path):
    """
    Translate spoken English audio to spoken Hokkien audio.

    Args:
        audio_path (str): The file path to the input English audio file.
        output_path (str): The file path where the translated Hokkien audio will be saved.

    Returns:
        str: The file path of the saved translated Hokkien audio.

    Raises:
        FileNotFoundError: If the input audio file does not exist.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'Input audio file not found at {audio_path}')

    # Load speech-to-speech translation model
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_s2ut_en-hk')
    model = models[0].cpu()
    cfg['task'].cpu = True

    # Generate translated text prediction
    generator = task.build_generator([model], cfg)
    audio, _ = torchaudio.load(audio_path)
    sample = S2THubInterface.get_model_input(task, audio)
    unit = S2THubInterface.get_prediction(task, model, generator, sample)

    # Load CodeHiFiGANVocoder model
    vocoder_cache_dir = snapshot_download('facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS')
    vocoder_dict = hub_utils.from_pretrained(
        vocoder_cache_dir,
        'model.pt',
        vocoder_cache_dir,
        archive_map=CodeHiFiGANVocoder.hub_models(),
        config_yaml='config.json',
        fp16=False,
        is_vocoder=True
    )
    vocoder = CodeHiFiGANVocoder(vocoder_dict['args']['model_path'][0], vocoder_dict['cfg'])

    # Convert translated text to speech
    tts_model = VocoderHubInterface(vocoder_dict['cfg'], vocoder)
    tts_sample = tts_model.get_model_input(unit)
    wav, sr = tts_model.get_prediction(tts_sample)

    # Save translated spoken Hokkien audio
    torchaudio.save(output_path, wav, sr)

    return output_path

# test_function_code --------------------

def test_translate_audio_en_to_hokkien():
    print("Testing started.")
    # Assuming 'sample_en_audio.wav' and 'output_hokkien_audio.wav' are valid paths
    input_audio_path = 'sample_en_audio.wav'
    expected_output_audio_path = 'output_hokkien_audio.wav'

    # Test case 1: Check if the function runs without error and returns correct output path
    print("Testing case [1/1] started.")
    output_audio_path = translate_audio_en_to_hokkien(input_audio_path, expected_output_audio_path)
    assert output_audio_path == expected_output_audio_path, f"Test case [1/1] failed: Expected output path to be {expected_output_audio_path}, got {output_audio_path}"
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_audio_en_to_hokkien()