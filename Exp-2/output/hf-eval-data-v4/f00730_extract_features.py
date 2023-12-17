# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def extract_features(source_code_text):
    """
    Extracts features from the given source code text using the UniXcoder model.

    Parameters:
        source_code_text (str): A string containing the source code from which to extract features.

    Returns:
        torch.Tensor: A tensor representing the feature matrix extracted from the source code.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('microsoft/unixcoder-base')
    model = AutoModel.from_pretrained('microsoft/unixcoder-base')

    # Tokenize the input source code text
    inputs = tokenizer(source_code_text, return_tensors='pt')

    # Get the model output
    outputs = model(**inputs)

    # Extract the feature matrix
    feature_matrix = outputs.last_hidden_state

    return feature_matrix

# test_function_code --------------------

def test_extract_features():
    print("Testing extract_features function.")
    
    # A sample source code for testing
    sample_source_code = "def hello_world():\n    print('Hello, world!')"

    # Expected shape of the feature matrix (depends on model and input length)
    expected_shape = (1, None, 768)  # The model outputs 768 features

    # Call the function
    feature_matrix = extract_features(sample_source_code)

    # Test if the output shape is correct
    assert feature_matrix.shape[:2] == expected_shape[:2], f"Test failed: Output shape {{feature_matrix.shape}} does not match expected shape {{expected_shape}}"

    print("Testing completed successfully.")

# Run the test
try:
    test_extract_features()
except Exception as e:
    print(f"A test case failed with an exception: {e}")