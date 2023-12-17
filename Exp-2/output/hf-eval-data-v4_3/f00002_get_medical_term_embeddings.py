# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def get_medical_term_embeddings(medical_term: str) -> torch.Tensor:
    """
    Generates embeddings for a given medical term using a pretrained model.

    Args:
        medical_term (str): The medical term for which to generate embeddings.

    Returns:
        torch.Tensor: The embeddings for the medical term.

    Raises:
        ValueError: If the medical term is not a string or is empty.
    """
    if not isinstance(medical_term, str) or not medical_term:
        raise ValueError('Medical term must be a non-empty string.')

    tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')
    model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')

    inputs = tokenizer(medical_term, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    return embeddings.squeeze()

# test_function_code --------------------

def test_get_medical_term_embeddings():
    print("Testing started.")

    # Test with a valid medical term
    print("Testing case [1/2] started.")
    valid_term = 'pneumonia'
    embeddings = get_medical_term_embeddings(valid_term)
    assert embeddings is not None, f"Test case [1/2] failed: Expected embeddings for the term '{valid_term}', got None."

    # Test with an invalid medical term
    print("Testing case [2/2] started.")
    invalid_term = ''
    try:
        _ = get_medical_term_embeddings(invalid_term)
        assert False, f"Test case [2/2] failed: Expected ValueError for the term '{invalid_term}'."
    except ValueError:
        assert True

    print("Testing finished.")

# call_test_function_line --------------------

test_get_medical_term_embeddings()