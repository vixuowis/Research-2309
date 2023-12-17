# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import HubertModel, Wav2Vec2Processor

# function_code --------------------

def extract_speech_features(audio_path):
    """
    Extract speech features from audio using the pretrained Hubert model.

    Parameters:
        audio_path (str): Path to the audio file.

    Returns:
        dict: A dictionary containing extracted features from the audio.
    """
    # Load the pretrained Hubert model
    hubert_model = HubertModel.from_pretrained('facebook/hubert-large-ll60k')
    # Load the Wav2Vec2 processor
    processor = Wav2Vec2Processor.from_pretrained('facebook/hubert-large-ll60k')

    # Load and preprocess the audio
    with open(audio_path, 'rb') as audio_file:
        audio_input = processor(audio_file.read(), sampling_rate=16000, return_tensors='pt')

    # Extract features using the Hubert model
    features = hubert_model(audio_input.input_values)
    return {'last_hidden_state': features.last_hidden_state}


# test_function_code --------------------

def test_extract_speech_features():
    print("Testing extract_speech_features function.")
    test_audio_path = 'test_audio.wav'

    # Test case: Extracting features from an audio file
    print("Test case: Extracting features from an audio file.")
    features = extract_speech_features(test_audio_path)
    assert 'last_hidden_state' in features, "Test failed: 'last_hidden_state' not in the extracted features."
    print("Test passed: Features successfully extracted.")

    print("Testing finished.")

# Run the test
test_extract_speech_features()
