from transformers import AutoTokenizer, AutoModel

def extract_features(source_code_text):
    """
    This function uses the 'microsoft/unixcoder-base' model from Hugging Face Transformers to tokenize the source code text and extract features.
    The feature matrix is derived as a matrix of embeddings using the model and tokenizer.
    
    Args:
    source_code_text (str): The source code text to be tokenized and for which features are to be extracted.
    
    Returns:
    torch.Tensor: The feature matrix derived from the source code text.
    """
    tokenizer = AutoTokenizer.from_pretrained('microsoft/unixcoder-base')
    model = AutoModel.from_pretrained('microsoft/unixcoder-base')
    inputs = tokenizer(source_code_text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state