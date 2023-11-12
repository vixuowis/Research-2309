# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def rank_search_results(query: str, passages: list) -> list:
    """
    Ranks the given passages based on their relevance to the given query using a pretrained model.

    Args:
        query (str): The search query.
        passages (list): The list of passages to be ranked.

    Returns:
        list: The list of passages ranked in descending order of relevance.
    """
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

    features = tokenizer([query] * len(passages), passages, padding=True, truncation=True, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits

    sorted_passages = sorted(zip(passages, scores.squeeze().tolist()), key=lambda x: x[1], reverse=True)
    return sorted_passages

# test_function_code --------------------

def test_rank_search_results():
    """
    Tests the rank_search_results function with some test cases.
    """
    query = 'How many people live in Berlin?'
    passages = [
        'Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.',
        'New York City is famous for the Metropolitan Museum of Art.',
        'Berlin is the capital of Germany and one of the 16 states of Germany.',
        'Berlin is known for its festivals, diverse architecture, nightlife, contemporary arts, and a high quality of living.'
    ]
    result = rank_search_results(query, passages)
    assert isinstance(result, list), 'The result should be a list.'
    assert len(result) == len(passages), 'The result should have the same length as the input passages.'
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result), 'Each item in the result should be a tuple with two elements.'
    assert all(isinstance(item[0], str) and isinstance(item[1], float) for item in result), 'Each item in the result should be a tuple with a string and a float.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_rank_search_results()