# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def rank_search_results(query, passages):
    """
    This function ranks the search results based on the relevance to the query.
    It uses the 'cross-encoder/ms-marco-MiniLM-L-6-v2' model from Hugging Face Transformers for sequence classification.

    Args:
        query (str): The search query.
        passages (list): A list of passages (search results).

    Returns:
        list: A list of tuples where each tuple contains a passage and its corresponding score. The list is sorted in descending order of scores.
    """
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

    features = tokenizer([query] * len(passages), passages, padding=True, truncation=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits

    sorted_passages = sorted(zip(passages, scores.squeeze().tolist()), key=lambda x: x[1], reverse=True)
    return sorted_passages

# test_function_code --------------------

def test_rank_search_results():
    """
    This function tests the rank_search_results function.
    It uses a sample query and passages for testing.
    """
    query = "Example search query"
    passages = [
        "passage 1",
        "passage 2",
        "passage 3"
    ]
    result = rank_search_results(query, passages)
    assert isinstance(result, list), "The result should be a list."
    assert all(isinstance(i, tuple) and len(i) == 2 for i in result), "Each item in the result should be a tuple with two elements."
    assert all(isinstance(i[0], str) and isinstance(i[1], float) for i in result), "Each tuple should contain a string and a float."

# call_test_function_code --------------------

test_rank_search_results()