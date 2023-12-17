# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def extract_biomedical_features(entity_names):
    """
    This function takes a list of biomedical entity names and extracts features
    using the pre-trained SapBERT model.
    
    Parameters:
        entity_names (list of str): A list of biomedical entity names.

    Returns:
        list of torch.Tensor: A list of [CLS] embeddings of features for each entity.
    """
    # Initialize tokenizer and model from Hugging Face Transformers
    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    
    # Process each entity name and return the [CLS] embeddings
    cls_embeddings = []
    for name in entity_names:
        inputs = tokenizer(name, return_tensors='pt')
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embeddings.append(cls_embedding)
    
    return cls_embeddings

# test_function_code --------------------

def test_extract_biomedical_features():
    print("Testing started.")
    # This is a hypothetical function to load test data, replace it with a real one if available.
    # dataset = load_dataset("biomedical_entity_names")
    sample_data = ["covid infection", "heart attack", "diabetes"]

    print("Testing case [1/1] started.")
    embeddings = extract_biomedical_features(sample_data)
    assert all([embedding.shape == (1, 768) for embedding in embeddings]), f"Test case [1/1] failed: Embeddings shape does not match expected shape."
    
    print("All test cases passed.")
    print("Testing finished.")

# Run the test function
test_extract_biomedical_features()