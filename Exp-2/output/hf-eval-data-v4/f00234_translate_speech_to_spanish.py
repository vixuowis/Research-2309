# requirements_file --------------------

!pip install -U fairseq torchaudio IPython.display huggingface_hub

# function_import --------------------

import torchaudio
import IPython.display as ipd
from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download

# function_code --------------------

def translate_speech_to_spanish(input_audio_file):
    # Load pre-trained models and task configuration from Hugging Face Hub
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub(
        'facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur')
    # Select the CPU for inference
    model = models[0].cpu()
    cfg['task'].cpu = True
    # Create a generator for speech-to-speech translation
    generator = task.build_generator([model], cfg)
    
    # Load the input English audio file
    audio, _ = torchaudio.load(input_audio_file)
    # Process the audio file and prepare it for translation
    sample = S2THubInterface.get_model_input(task, audio)
    # Perform the translation from English to Spanish
    translation_unit = S2THubInterface.get_prediction(task, model, generator, sample)

    # Load vocoder model for speech synthesis
    cache_dir = None
    cache_dir = snapshot_download('facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur', cache_dir=cache_dir)
    x = hub_utils.from_pretrained(cache_dir, 'model.pt', '.', archive_map=CodeHiFiGANVocoder.hub_models(), config_yaml='config.json', fp16=False, is_vocoder=True)
    
    # Setup the vocoder
    vocoder = CodeHiFiGANVocoder(x['args']['model_path'][0], x['model_cfg'])
    tts_model = VocoderHubInterface(x['model_cfg'], vocoder)
    tts_sample = tts_model.get_model_input(translation_unit)
    # Synthesize the translated audio
    wav, sr = tts_model.get_prediction(tts_sample)
    # Return the synthesized Spanish audio and its sample rate
    return wav, sr

# test_function_code --------------------

def test_translate_speech_to_spanish():
    # This is a placeholder for the actual audio file
    input_audio_file = 'test_audio_english.flac'
    print("Testing translation of English speech to Spanish.")
    
    # Run the translation function
    translated_wav, sample_rate = translate_speech_to_spanish(input_audio_file)
    
    # Check if the translation and synthesis were successful
    assert isinstance(translated_wav, torch.Tensor), "The translated audio should be a torch Tensor."
    assert sample_rate == 16000, "The sample rate should be 16 kHz."
    
    # Optionally, play the audio (commented out in a test environment)
    # ipd.Audio(translated_wav, rate=sample_rate)
    
    print("Translation to Spanish test passed.")
    
# Run the test
test_translate_speech_to_spanish()