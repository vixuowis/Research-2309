from transformers import AutoModel, AutoTokenizer


def extract_features_russian_text(text):
    """
    This function extracts features from a given Russian text using the 'DeepPavlov/rubert-base-cased' model.
    
    Parameters:
    text (str): The Russian text from which to extract features.
    
    Returns:
    torch.Tensor: The extracted features from the text.
    """
    # Create an instance of the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')
    
    # Convert the text message into input tokens
    inputs = tokenizer(text, return_tensors='pt')
    
    # Use the model to extract features from these tokens
    outputs = model(**inputs)
    
    # Return the extracted features
    return outputs.last_hidden_state