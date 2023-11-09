import torch
from transformers import BertModel, BertTokenizerFast

def get_sentence_embeddings(sentences):
    """
    This function takes a list of sentences in various languages and returns their embeddings using the LaBSE model.
    
    Args:
        sentences (list): A list of sentences. Each sentence is a string.
    
    Returns:
        torch.Tensor: A tensor containing the embeddings of the input sentences.
    
    Raises:
        ValueError: If the input is not a list or if any of the elements in the list is not a string.
    """
    if not isinstance(sentences, list) or not all(isinstance(sentence, str) for sentence in sentences):
        raise ValueError('Input should be a list of strings.')
    
    tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE')
    model = BertModel.from_pretrained('setu4993/LaBSE')
    model = model.eval()
    
    inputs = tokenizer(sentences, return_tensors='pt', padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.pooler_output