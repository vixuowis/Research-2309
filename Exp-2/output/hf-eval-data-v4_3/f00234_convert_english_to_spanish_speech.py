# requirements_file --------------------

import subprocess

requirements = ["fairseq", "torchaudio"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download
import torchaudio
import IPython.display as ipd


# function_code --------------------

def convert_english_to_spanish_speech(input_audio_file: str) -> IPython.display.Audio:
    """
    Convert English speech to Spanish speech using a trained Fairseq model.

    Args:
        input_audio_file (str): The path to the FLAC file containing English speech.

    Returns:
        IPython.display.Audio: An Audio object that can be played in an IPython environment.

    Raises:
        FileNotFoundError: If the input audio file does not exist.
    """
    # Verify the input audio file exists
    if not os.path.isfile(input_audio_file):
        raise FileNotFoundError(f"The audio file {input_audio_file} does not exist.")

    # Load the model and the task configuration
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub(
        'facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur'
    )
    model = models[0].cpu()
    cfg['task'].cpu = True

    # Load the English audio
    audio, _ = torchaudio.load(input_audio_file)
    sample = S2THubInterface.get_model_input(task, audio)

    # Generate the Spanish translation speech
    translation_unit = S2THubInterface.get_prediction(task, model, cfg['task'], sample)

    # Load the vocoder
    cache_dir = snapshot_download('facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur')
    x = hub_utils.from_pretrained(
        cache_dir,
        'model.pt',
        '.',
        archive_map=CodeHiFiGANVocoder.hub_models(),
        config_yaml='config.json',
        fp16=False,
        is_vocoder=True
    )
    vocoder = CodeHiFiGANVocoder(x['args']['model_path'][0], x['model_cfg'])
    tts_model = VocoderHubInterface(x['model_cfg'], vocoder)

    # Synthesize the Spanish speech
    tts_sample = tts_model.get_model_input(translation_unit)
    wav, sr = tts_model.get_prediction(tts_sample)

    # Return the synthesized audio
    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_convert_english_to_spanish_speech():
    print("Testing started.")
    # Path to a sample English audio file in FLAC format
    sample_audio_file = 'sample_english.flac'

    # Test case 1: Convert English to Spanish speech
    print("Testing case [1/1] started.")
    try:
        spanish_audio = convert_english_to_spanish_speech(sample_audio_file)
        assert isinstance(spanish_audio, ipd.Audio), "The output must be an IPython.display.Audio instance."
    except FileNotFoundError as e:
        assert False, f"Test case [1/1] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_convert_english_to_spanish_speech()