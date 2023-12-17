# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# function_code --------------------

def rank_passages_by_relevance(keyword, passages):
    """
    Rank given text passages based on their relevance to a specified keyword using
    a pretrained cross-encoder model.

    :param keyword: str - The keyword to search for within the passages.
    :param passages: list - A list of text passages to be ranked.
    :return: list - The passages sorted by relevance to the keyword.
    """
    model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    features = tokenizer([keyword] * len(passages), passages, padding=True, truncation=True, return_tensors='pt')

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
    sorted_indices = scores.argsort(descending=True)
    return [passages[i] for i in sorted_indices]

# test_function_code --------------------

def test_rank_passages_by_relevance():
    print("Testing started.")
    keyword = 'climate change'
    passages = [
        'Climate change is causing rising sea levels.',
        'Chocolate is the best dessert.',
        'Polar bears are affected by global warming.'
    ]

    expected_ranking = [
        'Climate change is causing rising sea levels.',
        'Polar bears are affected by global warming.',
        'Chocolate is the best dessert.'
    ]

    # Test case
    print("Testing relevance ranking.")
    ranked_passages = rank_passages_by_relevance(keyword, passages)
    assert ranked_passages == expected_ranking, "Test failed: The passages were not ranked correctly."

    print("Testing finished.")

# Run test function
test_rank_passages_by_relevance()