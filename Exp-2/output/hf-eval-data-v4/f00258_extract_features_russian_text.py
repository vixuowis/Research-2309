# requirements_file --------------------

!pip install -U transformers, torch

# function_import --------------------

from transformers import AutoModel, AutoTokenizer
import torch

# function_code --------------------

def extract_features_russian_text(text):
    """
    This function takes a Russian text string as input and uses the pre-trained model
    'DeepPavlov/rubert-base-cased' to extract features from the text.
    
    Args:
    text (str): Russian text from which to extract features.
    
    Returns:
    torch.Tensor: Extracted features from the text.
    """
    # Initialize the tokenizer and model from the pre-trained 'DeepPavlov/rubert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')

    # Tokenize the input text and convert to input IDs
    inputs = tokenizer(text, return_tensors='pt')

    # Generate the features using the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the last hidden states as the features
    features = outputs.last_hidden_state
    return features

# test_function_code --------------------

def test_extract_features_russian_text():
    print("Testing started.")
    sample_text = "Введите текст на русском языке здесь"  # Sample Russian text

    # Test case 1: Extracting features from sample Russian text
    print("Testing case [1/1] started.")
    features = extract_features_russian_text(sample_text)
    assert features is not None, "Test case failed: No features returned."
    assert isinstance(features, torch.Tensor), "Test case failed: The returned type is not a torch.Tensor."
    print("All tests passed.")
    print("Testing finished.")

# Run the test function
test_extract_features_russian_text()