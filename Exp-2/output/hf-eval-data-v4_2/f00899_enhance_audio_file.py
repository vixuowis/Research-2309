# requirements_file --------------------

!pip install -U torchaudio speechbrain

# function_import --------------------

import torchaudio
from speechbrain.pretrained import WaveformEnhancement

# function_code --------------------

def enhance_audio_file(input_filepath, output_filepath):
    """
    Enhances the audio quality of a given .wav file by reducing background noise using a pretrained model.

    Args:
        input_filepath (str): The path to the .wav file to be enhanced.
        output_filepath (str): The path where the enhanced .wav file will be saved.

    Returns:
        str: The path to the enhanced audio file.

    Raises:
        FileNotFoundError: If the input .wav file does not exist.
    """
    enhance_model = WaveformEnhancement.from_hparams(
        source="speechbrain/mtl-mimic-voicebank",
        savedir="pretrained_models/mtl-mimic-voicebank",
    )
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Input file does not exist: {input_filepath}")
    enhanced = enhance_model.enhance_file(input_filepath)
    torchaudio.save(output_filepath, enhanced.unsqueeze(0).cpu(), 16000)
    return output_filepath

# test_function_code --------------------

def test_enhance_audio_file():
    print("Testing started.")
    input_filepath = "test_input.wav"
    output_filepath = "test_output.wav"

    # Testing case 1: Verify file presence
    print("Testing case [1/3] started.")
    assert os.path.exists(input_filepath), f"Test case [1/3] failed: {input_filepath} does not exist."

    # Testing case 2: Enhancing and saving audio
    print("Testing case [2/3] started.")
    enhanced_file = enhance_audio_file(input_filepath, output_filepath)
    assert os.path.exists(enhanced_file), f"Test case [2/3] failed: {enhanced_file} file not created."

    # Testing case 3: Checking output file content is not None
    print("Testing case [3/3] started.")
    wave, sample_rate = torchaudio.load(output_filepath)
    assert wave is not None and sample_rate is not None, f"Test case [3/3] failed: Output file {output_filepath} has no content."
    print("Testing finished.")

# call_test_function_line --------------------

test_enhance_audio_file()