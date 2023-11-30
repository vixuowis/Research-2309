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
    tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")  # Tokenize input to match the trained model's tokenization
    medical_term_embedding = None
    
    if len(medical_term) > 0:
        model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")
        tokens = torch.tensor([tokenizer.encode(str(medical_term), max_length=512, pad_to_max_length=True)]) # Convert medical term into a tensor to be fed into the model for embedding creation.
        outputs = model(tokens)
        
        medical_term_embedding = outputs[1][0]  # Grab the first element of the output tuple which contains the embeddings of each input.
    
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