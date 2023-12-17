# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def rank_passages_by_keyword(keyword, passages):
    """
    Rank text passages based on their relevance to the given keyword.

    Args:
        keyword (str): The keyword to use for ranking the passages.
        passages (List[str]): List of passages to be ranked.

    Returns:
        List[str]: List of passages sorted by relevance to the keyword.

    Raises:
        ValueError: If keyword is not provided.
        ValueError: If no passages are provided.
    """
    if not keyword:
        raise ValueError('Keyword is required for ranking passages.')
    if not passages:
        raise ValueError('At least one passage is required for ranking.')

    # Load the model and tokenizer
    model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the query and passages
    features = tokenizer([keyword] * len(passages), passages, padding=True, truncation=True, return_tensors='pt')

    # Score the passages
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits

    # Sort passages based on scores and return the sorted list
    sorted_passages = [pass for _, pass in sorted(zip(scores.squeeze().tolist(), passages), reverse=True)]
    return sorted_passages

# test_function_code --------------------

def test_rank_passages_by_keyword():
    print("Testing started.")
    keyword = 'science'
    passages = [
        'Science is both a body of knowledge and a process.',
        'In school, science may sometimes seem like a collection of isolated and static facts.',
        'Science is exciting. It is a way of discovering what’s in the universe and how those things work.'
    ]

    # Test case 1: Check if the function ranks the passages correctly
    print("Testing case [1/1] started.")
    ranked_passages = rank_passages_by_keyword(keyword, passages)
    assert ranked_passages == sorted(passages, reverse=True), f"Test case [1/1] failed: Expected {sorted(passages, reverse=True)}, got {ranked_passages}"
    print("Testing finished.")

# call_test_function_line --------------------

test_rank_passages_by_keyword()