# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------


def denoise_audio(input_audio_path, output_audio_path):
    """
    Denoises an audio file using a pre-trained Sepformer model.

    Parameters:
        input_audio_path (str): The file path of the input noisy audio.
        output_audio_path (str): The file path where the cleaned audio will be saved.

    Returns:
        None: The enhanced audio is saved to 'output_audio_path'.
    """
    # Load the pre-trained Sepformer model from Hugging Face Transformers
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    
    # Denoise the audio file
    est_sources = model.separate_file(path=input_audio_path)
    
    # Save the enhanced audio file
    torchaudio.save(output_audio_path, est_sources[:, :, 0].detach().cpu(), 16000)


# test_function_code --------------------


def test_denoise_audio():
    print("Testing started.")
    input_audio_path = 'input_audio_file.wav'  # You should provide a sample noisy audio file
    output_audio_path = 'enhanced_audio_file.wav'  # Output path for the enhanced audio

    # Test case: Check if the output file exists after running the denoising function
    print("Testing case [1/1] started.")
    denoise_audio(input_audio_path, output_audio_path)
    assert os.path.exists(output_audio_path), f"Test case failed: no output file found at {output_audio_path}"
    print("Testing case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_denoise_audio()
