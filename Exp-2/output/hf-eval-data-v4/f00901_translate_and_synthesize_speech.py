# requirements_file --------------------

!pip install -U fairseq torchaudio huggingface_hub

# function_import --------------------

from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download
import torchaudio


# function_code --------------------

def translate_and_synthesize_speech(input_audio_path, output_audio_path):
    """
    Translate an audio file from one language to another and synthesize the speech.

    Parameters:
        input_audio_path (str): The file path for the input audio to be translated.
        output_audio_path (str): The file path to save the synthesized output audio.
    """
    # Load the speech-to-speech translation model
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_unity_hk-en')
    model = models[0].cpu()
    generator = task.build_generator([model], cfg)

    # Preprocess the input audio file
    audio, _ = torchaudio.load(input_audio_path)
    sample = S2THubInterface.get_model_input(task, audio)
    translated_speech = S2THubInterface.get_prediction(task, model, generator, sample)

    # Load the HiFi-GAN vocoder model for speech synthesis
    cache_dir = snapshot_download('facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur')
    vocoder_args, vocoder_cfg = hub_utils.from_pretrained(cache_dir, is_vocoder=True)
    vocoder = CodeHiFiGANVocoder(vocoder_args['model_path'][0], vocoder_cfg)
    tts_model = VocoderHubInterface(vocoder_cfg, vocoder)

    # Generate synthesized speech from the translated text
    tts_sample = tts_model.get_model_input(translated_speech)
    synthesized_speech, sample_rate = tts_model.get_prediction(tts_sample)

    # Save synthesized speech as an audio file
    torchaudio.save(output_audio_path, synthesized_speech, sample_rate)


# test_function_code --------------------

def test_translate_and_synthesize_speech():
    print("Testing started.")

    input_audio_path = 'test_input.wav'
    output_audio_path = 'test_output.wav'

    # Assume test_input.wav exists and contains audio data
    # Call the translate_and_synthesize_speech function
    translate_and_synthesize_speech(input_audio_path, output_audio_path)

    # Check the output file exists
    assert os.path.exists(output_audio_path), "Output audio file was not created."

    # Load the output file to check if it's not empty
    output_audio, _ = torchaudio.load(output_audio_path)
    assert output_audio.shape[1] > 0, "Output audio file is empty."

    print("Testing finished.")

# Run the test
test_translate_and_synthesize_speech()
