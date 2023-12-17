# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def extract_entity_relations(text):
    """
    Extract positional relations between biomedical entities in the given text using
    the SapBERT model from Hugging Face Transformers.

    Parameters:
    text (str): Biomedical text from which to extract entity relations.

    Returns:
    torch.Tensor: The [CLS] embedding from SapBERT model representing positional relations.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt')

    # Get model outputs
    outputs = model(**inputs)

    # Extract [CLS] embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding

# test_function_code --------------------

def test_extract_entity_relations():

    print("Testing extract_entity_relations function.")

    # Test case 1: Simple biomedical text
    text_1 = 'covid infection in lungs'
    embedding_1 = extract_entity_relations(text_1)
    assert embedding_1.shape == (1, 768), f"Test case 1 failed: Unexpected embedding shape {embedding_1.shape}"

    # Test case 2: Another biomedical text
    text_2 = 'breast cancer treatment options'
    embedding_2 = extract_entity_relations(text_2)
    assert embedding_2.shape == (1, 768), f"Test case 2 failed: Unexpected embedding shape {embedding_2.shape}"

    # Test case 3: Empty text
    text_3 = ''
    embedding_3 = extract_entity_relations(text_3)
    assert embedding_3.shape == (1, 768), f"Test case 3 failed: Unexpected embedding shape {embedding_3.shape}"

    print("All test cases passed.")