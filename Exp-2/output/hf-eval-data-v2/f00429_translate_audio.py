# function_import --------------------

import os
import torchaudio
from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download

# function_code --------------------

def translate_audio(input_audio_path: str, output_audio_path: str = 'translated_hokkien_audio.wav') -> None:
    """
    Translates spoken English audio to spoken Hokkien audio.

    Args:
        input_audio_path (str): Path to the input English audio file.
        output_audio_path (str, optional): Path to save the translated Hokkien audio. Defaults to 'translated_hokkien_audio.wav'.

    Returns:
        None
    """
    # Load speech-to-speech translation model
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_s2ut_en-hk')
    model = models[0].cpu()
    cfg['task'].cpu = True

    # Generate translated text prediction
    generator = task.build_generator([model], cfg)
    audio, _ = torchaudio.load(input_audio_path)
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
    torchaudio.save(output_audio_path, wav, sr)

# test_function_code --------------------

def test_translate_audio():
    """
    Test the function translate_audio.
    """
    # Define input and output paths
    input_audio_path = '/path/to/an/english/audio/file'
    output_audio_path = 'translated_hokkien_audio.wav'

    # Call the function
    translate_audio(input_audio_path, output_audio_path)

    # Check if the output file is created
    assert os.path.exists(output_audio_path), 'Output audio file not found.'

# call_test_function_code --------------------

test_translate_audio()