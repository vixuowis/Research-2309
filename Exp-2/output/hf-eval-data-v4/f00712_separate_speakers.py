# requirements_file --------------------

!pip install -U asteroid transformers torch soundfile

# function_import --------------------

from transformers import Asteroid
import torch
import soundfile as sf

# function_code --------------------

def separate_speakers(audio_path: str, output_path: str) -> None:
    """
    Separates overlapping speakers in an audio file and writes the separated audio to a new file.

    Args:
    audio_path (str): The file path to the mixed audio with overlapping speakers.
    output_path (str): The file path to write the separated audio.
    """
    # Load the ConvTasNet_Libri2Mix_sepclean_16k model
    model = Asteroid('JorisCos/ConvTasNet_Libri2Mix_sepclean_16k')
    # Read the audio file
    mixed_audio, sample_rate = sf.read(audio_path)
    # Convert the audio to a tensor
    mixed_audio_tensor = torch.tensor(mixed_audio)
    # Use the model to separate speakers
    separated_audio_tensor = model(mixed_audio_tensor)
    # Convert tensor to numpy array
    separated_audio = separated_audio_tensor.numpy()
    # Write the separated audio to a new file
    sf.write(output_path, separated_audio, sample_rate)

# test_function_code --------------------

def test_separate_speakers():
    print("Testing started.")
    # Path to a sample audio file with overlapping speakers
    sample_audio_path = 'path_to_sample_mixed_audio.wav'
    # Path to write the output
    output_audio_path = 'path_to_output_separated_audio.wav'

    # Perform the speaker separation
    separate_speakers(sample_audio_path, output_audio_path)

    # Read the output file to confirm it was written
    try:
        separated_audio, _ = sf.read(output_audio_path)
        print("Output file created successfully.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Testing finished.")

# Run the test function
test_separate_speakers()