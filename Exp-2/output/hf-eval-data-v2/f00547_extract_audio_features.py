# function_import --------------------

from transformers import HubertModel

# function_code --------------------

def extract_audio_features(audio_sample):
    """
    This function uses the Hubert-large-ll60k model from Hugging Face Transformers to extract features from an audio sample.
    
    Args:
        audio_sample (str): Path to the audio sample file.
    
    Returns:
        torch.Tensor: The extracted features from the audio sample.
    
    Raises:
        Exception: If the audio sample file does not exist.
    """
    # Import the necessary class from the transformers package
    from transformers import HubertModel
    
    # Load the pre-trained model 'facebook/hubert-large-ll60k'
    hubert = HubertModel.from_pretrained('facebook/hubert-large-ll60k')
    
    # Check if the audio sample file exists
    if not os.path.exists(audio_sample):
        raise Exception(f'The audio sample file {audio_sample} does not exist.')
    
    # Load the audio sample
    audio_data = load_audio_sample(audio_sample)
    
    # Use the model for feature extraction on the audio sample
    features = hubert(audio_data)
    
    return features

# test_function_code --------------------

def test_extract_audio_features():
    """
    This function tests the extract_audio_features function by using a sample audio file.
    """
    # Define the path to the sample audio file
    sample_audio_file = 'sample_audio_file.wav'
    
    # Call the extract_audio_features function
    features = extract_audio_features(sample_audio_file)
    
    # Check if the returned object is a torch.Tensor
    assert isinstance(features, torch.Tensor), 'The returned object is not a torch.Tensor.'

# call_test_function_code --------------------

test_extract_audio_features()