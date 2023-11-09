from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def rank_passages(query, passages):
    '''
    This function ranks text passages based on their importance regarding a given query.
    It uses the pretrained model 'cross-encoder/ms-marco-MiniLM-L-12-v2' from Hugging Face Transformers.
    
    Parameters:
    query (str): The query to rank the passages.
    passages (list): The list of passages to be ranked.
    
    Returns:
    list: The list of passages ranked in decreasing order of importance.
    '''
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    features = tokenizer(query, passages, padding=True, truncation=True, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
    sorted_passages = [passage for _, passage in sorted(zip(scores, passages), reverse=True)]
    return sorted_passages