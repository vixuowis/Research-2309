# requirements_file --------------------

!pip install -U fairseq hub_utils huggingface_hub IPython torchaudio

# function_import --------------------

from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download
import torchaudio
import IPython.display as ipd

# function_code --------------------

def convert_text_to_speech(text, language='en'):
    """
    Convert text input to speech audio in a specified language.

    Args:
        text (str): The text to be converted to speech.
        language (str): The language of the text (default is English).

    Returns:
        An IPython.display.Audio object with the generated speech audio.

    Raises:
        ValueError: If the language specified is not supported.
    """
    # List of supported languages
    supported_languages = ['en', 'es', 'fr', 'it']

    if language not in supported_languages:
        raise ValueError(f'Language {language} is not supported.')

    # Load pretrained models and configs from huggingface hub
    model_id = 'facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur'
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(model_id)

    # Convert text to audio sample
    sample = S2THubInterface.get_model_input(task, text, language)

    # Generate speech audio unit
    unit = S2THubInterface.get_prediction(task, models[0], task.build_generator([models[0]], cfg), sample)

    # Vocoder processing to get human-like speech
    vocoder_path = snapshot_download(model_id)
    vocoder_cfg = {'model_path': [f'{vocoder_path}/vocoder_model.pt']}
    vocoder = CodeHiFiGANVocoder.from_pretrained(vocoder_cfg)
    tts_model = VocoderHubInterface(vocoder_cfg, vocoder)

    # Prepare TTS model input and generate speech audio
    tts_sample = tts_model.get_model_input(unit)
    wav, sr = tts_model.get_prediction(tts_sample)

    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_convert_text_to_speech():
    print("Testing started.")

    # Test case 1: Convert English text to speech
    print("Testing case [1/3] started.")
    audio = convert_text_to_speech('Hello, this is a test.', 'en')
    assert isinstance(audio, ipd.Audio), "Test case [1/3] failed: The returned object is not an Audio object."

    # Test case 2: Convert Spanish text to speech
    print("Testing case [2/3] started.")
    audio = convert_text_to_speech('Hola, esto es una prueba.', 'es')
    assert isinstance(audio, ipd.Audio), "Test case [2/3] failed: The returned object is not an Audio object."

    # Test case 3: Attempt to convert unsupported language text to speech
    print("Testing case [3/3] started.")
    try:
        convert_text_to_speech('This is unsupported language test.', 'de')
        assert False, "Test case [3/3] failed: ValueError exception was expected."
    except ValueError as e:
        assert str(e) == 'Language de is not supported.', "Test case [3/3] failed: Incorrect error message."

    print("Testing finished.")

# call_test_function_line --------------------

test_convert_text_to_speech()