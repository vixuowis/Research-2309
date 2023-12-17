# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features(input_text_or_code):
    """
    This function uses the CodeBERT model to extract features from a given text or code snippet.
    
    Parameters:
    input_text_or_code (str): A string containing the text or code to extract features from.

    Returns:
    torch.Tensor: The embeddings representing features of the input text or code.
    """
    # Load the pre-trained CodeBERT model
    model = AutoModel.from_pretrained('microsoft/codebert-base')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    
    inputs = tokenizer(input_text_or_code, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    
    return embeddings

# test_function_code --------------------

def test_extract_features():
    import torch
    print("Testing started.")
    sample_text = "Extract features from this text using CodeBERT"
    sample_code = "def foo(bar): return bar * 2"
    text_embeddings = extract_features(sample_text)
    assert isinstance(text_embeddings, torch.Tensor), f"Test case [1/2] failed: Expected output type torch.Tensor, got {type(text_embeddings)}"
    code_embeddings = extract_features(sample_code)
    assert isinstance(code_embeddings, torch.Tensor), f"Test case [2/2] failed: Expected output type torch.Tensor, got {type(code_embeddings)}"
    print("Testing finished.")

test_extract_features()