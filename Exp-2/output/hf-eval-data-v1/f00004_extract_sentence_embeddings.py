from transformers import AutoModel, AutoTokenizer


def extract_sentence_embeddings(input_text):
    """
    This function takes a sentence as input and returns its embedding using the LaBSE model.
    The LaBSE model is a language-agnostic model for extracting sentence embeddings in multiple languages.
    
    Parameters:
    input_text (str): The sentence to be encoded.
    
    Returns:
    sentence_embedding (tensor): The embedding of the input sentence.
    """
    # Load the pre-trained LaBSE model and its tokenizer
    model = AutoModel.from_pretrained('rasa/LaBSE')
    tokenizer = AutoTokenizer.from_pretrained('rasa/LaBSE')
    
    # Encode the input sentence
    encoded_input = tokenizer(input_text, return_tensors='pt')
    
    # Extract the sentence embedding
    embeddings = model(**encoded_input)
    sentence_embedding = embeddings.pooler_output
    
    return sentence_embedding