# function_import --------------------

import torchaudio
from speechbrain.pretrained import WaveformEnhancement

# function_code --------------------

def enhance_audio(input_audio_file: str, output_audio_file: str) -> None:
    """
    Enhances the quality of an audio file by reducing background noise.

    Args:
        input_audio_file (str): The path to the input audio file that needs enhancement.
        output_audio_file (str): The path where the enhanced audio file will be saved.

    Returns:
        None
    """
    enhance_model = WaveformEnhancement.from_hparams(
        source="speechbrain/mtl-mimic-voicebank",
        savedir="pretrained_models/mtl-mimic-voicebank",
    )
    enhanced = enhance_model.enhance_file(input_audio_file)
    torchaudio.save(output_audio_file, enhanced.unsqueeze(0).cpu(), 16000)

# test_function_code --------------------

def test_enhance_audio():
    """
    Tests the enhance_audio function by enhancing a sample audio file and checking if the output file is created.
    """
    input_audio_file = 'test_input.wav'
    output_audio_file = 'test_output.wav'
    enhance_audio(input_audio_file, output_audio_file)
    assert os.path.exists(output_audio_file), 'Output file not created.'

# call_test_function_code --------------------

test_enhance_audio()