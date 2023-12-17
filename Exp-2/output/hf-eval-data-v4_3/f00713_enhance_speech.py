# requirements_file --------------------

import subprocess

requirements = ["speechbrain", "torchaudio"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_speech(input_audio_path: str, output_audio_path: str) -> None:
    """
    Enhances the speech by reducing noise from an audio file using a pre-trained SepFormer model.

    Args:
        input_audio_path (str): The file path to the input audio file to be processed.
        output_audio_path (str): The file path where the enhanced audio file will be saved.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input_audio_path does not exist.
        RuntimeError: If the model fails to process the audio.
    """
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement',
                                   savedir='pretrained_models/sepformer-wham16k-enhancement')
    est_sources = model.separate_file(path=input_audio_path)
    torchaudio.save(output_audio_path, est_sources[:, :, 0].detach().cpu(), 16000)

# test_function_code --------------------

def test_enhance_speech():
    print("Testing started.")
    input_audio_path = 'speechbrain/sepformer-wham16k-enhancement/example_wham16k.wav'
    output_audio_path = 'enhanced_wham16k.wav'

    # Testing case 1
    print("Testing case [1/1] started.")
    enhance_speech(input_audio_path, output_audio_path)
    assert os.path.exists(output_audio_path), f"Test case [1/1] failed: '{output_audio_path}' not found."
    print("Testing finished.")

# call_test_function_line --------------------

test_enhance_speech()