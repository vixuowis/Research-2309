# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def rank_search_results(query, passages):
    """
    Uses the Hugging Face Transformers model to rank a list of passages based on their relevance to a given query.

    Args:
        query (str): The search query string.
        passages (list): A list of passages (str) to be ranked.

    Returns:
        list: A list of tuples, each containing a passage and its associated score, sorted in descending order of score.

    Raises:
        ValueError: If the query or passages are not provided.
    """
    if not query or not passages:
        raise ValueError('Query and passages are required for ranking.')
    
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    features = tokenizer([query] * len(passages), passages, padding=True, truncation=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
    
    sorted_passages = sorted(zip(passages, scores.squeeze().tolist()), key=lambda x: x[1], reverse=True)
    return sorted_passages

# test_function_code --------------------

def test_rank_search_results():
    print("Testing started.")
    
    # Test case 1: Query with multiple passages
    print("Testing case [1/2] started.")
    query1 = "How to bake a cake"
    passages1 = [
        "Follow these steps to bake a cake.",
        "Learn to cook Italian cuisine.",
        "Cake baking instructions and tips."
    ]
    results1 = rank_search_results(query1, passages1)
    assert len(results1) == len(passages1), f"Test case [1/2] failed: Incorrect number of ranked passages returned."
    
    # Test case 2: No query or passages provided
    print("Testing case [2/2] started.")
    try:
        rank_search_results('', [])
        assert False, "Test case [2/2] failed: ValueError not raised for empty input."
    except ValueError as e:
        assert str(e) == 'Query and passages are required for ranking.', f"Test case [2/2] failed: Incorrect error message."
    
    print("Testing finished.")

# call_test_function_line --------------------

test_rank_search_results()