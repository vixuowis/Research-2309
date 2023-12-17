# requirements_file --------------------

import subprocess

requirements = ["transformers", "scipy", "numpy"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModelForSpeechClassification, Wav2Vec2Processor

# function_code --------------------

def identify_language_in_audio(audio_path: str) -> str:
    """Identify the language spoken in the audio file.

    Args:
        audio_path: Path to the audio file (in .wav format).

    Returns:
        The ISO language code of the language identified in the audio.

    Raises:
        FileNotFoundError: If the audio_path does not exist.
        ValueError: If the audio file format is not supported.
    """
    import os
    from scipy.io import wavfile

    # Ensure the audio file exists and is a .wav file
    if not os.path.exists(audio_path) or not audio_path.lower().endswith('.wav'):
        raise FileNotFoundError('The audio file does not exist or is not a .wav file.')

    # Load the audio file
    sr, audio = wavfile.read(audio_path)

    # Initialize the model and processor
    model = AutoModelForSpeechClassification.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')
    processor = Wav2Vec2Processor.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')

    # Preprocess the audio
    input_values = processor(audio, sampling_rate=sr, return_tensors='pt').input_values

    # Predict the language
    with torch.no_grad():
        logits = model(input_values).logits

    # Extract the predicted language code
    predicted_id = torch.argmax(logits, dim=-1)
    language_code = model.config.id2label[predicted_id.item()]

    return language_code

# test_function_code --------------------

def test_identify_language_in_audio():
    import tempfile
    import numpy as np
    from scipy.io.wavfile import write

    print('Testing started.')
    # Generate dummy audio data (1 sec of silence)
    sample_rate = 16000
    audio_data = np.zeros(sample_rate)

    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
        write(tmp.name, sample_rate, audio_data)

        # Test case 1: Identify language in silent audio (should return 'unknown')
        print('Testing case [1/1] started.')
        language_code = identify_language_in_audio(tmp.name)
        assert language_code == 'unknown', f'Test case [1/1] failed: Expected unknown, got {language_code}'

    print('Testing finished.')

# call_test_function_line --------------------

test_identify_language_in_audio()