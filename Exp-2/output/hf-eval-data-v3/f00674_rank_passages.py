# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def rank_passages(query: str, passages: list) -> list:
    """
    Ranks text passages based on their importance regarding a given keyword.

    Args:
        query (str): The keyword to search for.
        passages (list): The list of text passages to rank.

    Returns:
        list: The list of passages sorted in decreasing order of importance.

    Raises:
        OSError: If there is an error in loading the pretrained model or tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    features = tokenizer(query, passages, padding=True, truncation=True, return_tensors='pt')

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
   
    # Sort passages based on scores
    sorted_passages = [passage for _, passage in sorted(zip(scores, passages), reverse=True)]
    return sorted_passages

# test_function_code --------------------

def test_rank_passages():
    """
    Tests the rank_passages function with some test cases.
    """
    query = 'How many people live in Berlin?'
    passages = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
    result = rank_passages(query, passages)
    assert isinstance(result, list), 'The result should be a list.'
    assert len(result) == len(passages), 'The length of the result should be equal to the length of the passages.'
    assert result[0] == passages[0], 'The first passage should be the most relevant one.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_rank_passages()