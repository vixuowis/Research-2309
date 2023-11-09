# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def rank_passages(query, passages):
    """
    This function ranks text passages based on their importance regarding a given keyword.
    It uses the pretrained model 'cross-encoder/ms-marco-MiniLM-L-12-v2' from Hugging Face Transformers.

    Args:
        query (str): The keyword to search for.
        passages (list): The list of passages to rank.

    Returns:
        list: The passages sorted in decreasing order of importance.
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
    This function tests the rank_passages function.
    It uses a sample query and passages, and checks if the function returns a list.
    """
    query = 'How many people live in Berlin?'
    passages = ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.']
    result = rank_passages(query, passages)
    assert isinstance(result, list), 'The function should return a list.'
    assert len(result) == len(passages), 'The function should return a list of the same length as the input passages.'

# call_test_function_code --------------------

test_rank_passages()