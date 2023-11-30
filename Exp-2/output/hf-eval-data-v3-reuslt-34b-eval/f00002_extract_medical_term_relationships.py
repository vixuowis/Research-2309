# function_import --------------------

from transformers import AutoTokenizer, AutoModel
import torch

# function_code --------------------

def extract_medical_term_relationships(medical_term):
    """
    This function uses the pretrained model 'GanjinZero/UMLSBert_ENG' from Hugging Face Transformers to find relationships between medical terms.
    It converts the medical terms into embeddings (dense vectors) which can be compared to find similarities and relationships.

    Args:
        medical_term (str): The medical term to be converted into an embedding.

    Returns:
        torch.Tensor: The embedding of the input medical term.
    """    
    tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
    model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG").to(device='cuda')
    encoded_input = tokenizer(medical_term, return_tensors="pt").to(device='cuda')
    output = model(**encoded_input)
    
    medical_term_embedding = output.last_hidden_state[0][0] # embedding is a tensor of shape (768,)
    
    return medical_term_embedding

# test_function_code --------------------

def test_extract_medical_term_relationships():
    """
    This function tests the 'extract_medical_term_relationships' function with different medical terms.
    It asserts that the embeddings for different terms should not be the same.
    """
    embedding1 = extract_medical_term_relationships('Cancer')
    embedding2 = extract_medical_term_relationships('Diabetes')

    assert not torch.equal(embedding1, embedding2), 'Embeddings for different terms should not be the same.'

    embedding3 = extract_medical_term_relationships('Hypertension')
    embedding4 = extract_medical_term_relationships('Asthma')

    assert not torch.equal(embedding3, embedding4), 'Embeddings for different terms should not be the same.'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_extract_medical_term_relationships()