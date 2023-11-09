from transformers import AutoModel, AutoTokenizer


def extract_features(text):
    """
    This function uses the pre-trained ConvBERT model 'YituTech/conv-bert-base' from Hugging Face Transformers
    to extract features from the given text.

    Parameters:
    text (str): The text from which to extract features.

    Returns:
    torch.Tensor: The extracted features.
    """
    # Load the pre-trained ConvBERT model
    conv_bert_model = AutoModel.from_pretrained('YituTech/conv-bert-base')
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('YituTech/conv-bert-base')
    # Tokenize the text
    input_tokens = tokenizer.encode(text, return_tensors='pt')
    # Extract features
    features = conv_bert_model(**input_tokens).last_hidden_state
    return features