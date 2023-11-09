from transformers import AutoModel, AutoTokenizer

# Function to extract features using BioBERT

def extract_features_biobert(text):
    """
    This function takes in biomedical text as input and uses the BioBERT model to extract features.
    BioBERT is a pre-trained biomedical language representation model for biomedical text mining tasks such as biomedical named entity recognition, relation extraction, and question answering.
    
    Args:
    text (str): The biomedical text from which to extract features.
    
    Returns:
    tensor: The extracted features in the form of a tensor.
    """
    # Load the BioBERT model
    model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt')
    
    # Extract features
    outputs = model(**inputs)
    
    return outputs.last_hidden_state