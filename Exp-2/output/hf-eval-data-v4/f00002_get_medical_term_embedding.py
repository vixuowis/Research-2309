# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModel
import torch

# function_code --------------------

def get_medical_term_embedding(medical_term):
    """
    Convert a medical term into an embedding vector using the pre-trained UMLSBert_ENG model.

    Parameters:
    - medical_term: str - A medical term to be converted into an embedding.

    Returns:
    - embeddings: torch.Tensor - The embedding vector for the given medical term.
    """

    # Initialize the tokenizer and model from the pre-trained UMLSBert_ENG
    tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')
    model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')

    # Tokenize the medical term and convert to model input format
    inputs = tokenizer(medical_term, return_tensors="pt")
    
    # Pass the input through the model to get the embeddings
    with torch.no_grad():  # disable gradient calculation for inference
        outputs = model(**inputs)

    # Extract the embeddings for the last layer
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings

# test_function_code --------------------

import torch

def test_get_medical_term_embedding():
    print("Testing started.")
    # Sample medical terms for testing
    sample_medical_terms = ["aspirin", "myocardial infarction", "leukemia"]

    # Test cases
    for i, term in enumerate(sample_medical_terms):
        print(f"Testing case [{i+1}/{len(sample_medical_terms)}] started.")
        embeddings = get_medical_term_embedding(term)
        assert isinstance(embeddings, torch.Tensor), f"Test case [{i+1}/{len(sample_medical_terms)}] failed: The output is not a torch.Tensor."
        assert embeddings.shape == (1, 768), f"Test case [{i+1}/{len(sample_medical_terms)}] failed: The output shape is not correct."
        print(f"Testing case [{i+1}/{len(sample_medical_terms)}] successfully completed.")

    print("Testing finished.")

# Running the test function
test_get_medical_term_embedding()