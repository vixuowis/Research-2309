# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import HubertModel

# function_code --------------------

def extract_features_from_audio(audio_data):
    """
    Extracts features from an audio sample using Hubert-large-ll60k model.

    Args:
        audio_data (Tensor): A tensor containing the raw audio data to be processed.

    Returns:
        Tensor: A tensor containing the extracted audio features.

    Raises:
        ValueError: If 'audio_data' is not a tensor.

    """
    if not isinstance(audio_data, Tensor):
        raise ValueError("'audio_data' must be a tensor.")
    hubert = HubertModel.from_pretrained('facebook/hubert-large-ll60k')
    # Assuming 'audio_data' is already the correct format expected by the model
    features = hubert(audio_data)
    return features.last_hidden_state

# test_function_code --------------------

def test_extract_features_from_audio():
    print("Testing started.")
    # Assuming we have a function called 'load_sample_audio' that loads a sample audio data as a tensor
    audio_data, expected_features = load_sample_audio('sample_audio_path')

    # Test using a sample audio tensor
    print("Testing case [1/1] started.")
    extracted_features = extract_features_from_audio(audio_data)
    assert extracted_features.shape == expected_features.shape, f"Test case [1/1] failed: The extracted features shape {extracted_features.shape} does not match the expected shape {expected_features.shape}."
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_features_from_audio()