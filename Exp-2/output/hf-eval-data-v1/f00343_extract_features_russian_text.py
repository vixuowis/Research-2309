from transformers import AutoModel, AutoTokenizer


def extract_features_russian_text(input_text):
    """
    This function extracts features from Russian text using the pre-trained model 'DeepPavlov/rubert-base-cased'
    from the Hugging Face Transformers library.
    
    Parameters:
    input_text (str): The Russian text from which to extract features.
    
    Returns:
    torch.Tensor: The extracted features from the input text.
    """
    # Load the pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')
    
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Use the model to extract features from the tokenized input
    outputs = model(**inputs)
    
    # Return the extracted features
    return outputs.last_hidden_state