from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def rank_search_results(query, passages):
    """
    This function ranks search results based on their relevance to a given query.
    It uses the 'cross-encoder/ms-marco-MiniLM-L-6-v2' model from Hugging Face Transformers, which is trained for Information Retrieval tasks.

    Args:
        query (str): The search query.
        passages (list): A list of passages (documents) to be ranked.

    Returns:
        list: A list of tuples, where each tuple contains a passage and its corresponding score. The list is sorted in descending order of scores.
    """
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

    features = tokenizer([query] * len(passages), passages, padding=True, truncation=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits

    sorted_passages = sorted(zip(passages, scores.squeeze().tolist()), key=lambda x: x[1], reverse=True)
    return sorted_passages