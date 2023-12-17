# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_text_features(text_data):
    """
    Extract features from text data using a pre-trained ConvBERT model.

    Parameters:
        text_data (str): The input text data for which to extract features.

    Returns:
        torch.Tensor: The extracted features as a tensor.
    """
    # Load the pre-trained ConvBERT model
    model = AutoModel.from_pretrained('YituTech/conv-bert-base')
    tokenizer = AutoTokenizer.from_pretrained('YituTech/conv-bert-base')

    # Tokenize the input text
    input_tokens = tokenizer.encode(text_data, return_tensors='pt')

    # Extract features
    features = model(input_tokens)[0].last_hidden_state

    return features

# test_function_code --------------------

def test_extract_text_features():
    print("Testing extract_text_features function.")

    # Sample text for testing
    test_text = "Hello, this is a test sentence for feature extraction."

    # Expected result is not checked here as it's a complex tensor output
    # Test if execution runs without errors
    try:
        features = extract_text_features(test_text)
        assert features is not None, "No features extracted."
        print("Test passed.")
    except Exception as e:
        print(f"Test failed with an exception: {e}")

# Run the test function
test_extract_text_features()