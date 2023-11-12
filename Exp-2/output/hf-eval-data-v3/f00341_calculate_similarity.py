# function_import --------------------

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_similarity(review1: str, review2: str) -> float:
    """
    Calculate the similarity between two book reviews using a pretrained model.

    Args:
        review1 (str): The first book review.
        review2 (str): The second book review.

    Returns:
        float: The similarity score between the two reviews. The score is in the range of [-1, 1], with higher scores indicating more similarity.
    """
    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    model = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    input_tensors = tokenizer([review1, review2], return_tensors='pt', padding=True, truncation=True)
    embeddings = model(**input_tensors).pooler_output
    similarity_score = cosine_similarity(embeddings[0].detach().numpy().reshape(1, -1), embeddings[1].detach().numpy().reshape(1, -1))[0][0]
    return similarity_score

# test_function_code --------------------

def test_calculate_similarity():
    review1 = 'This book is amazing!'
    review2 = 'I love this book!'
    review3 = 'This book is not good.'
    assert 0.7 <= calculate_similarity(review1, review2) <= 1.0
    assert -1.0 <= calculate_similarity(review1, review3) <= 0.3
    return 'All Tests Passed'

# call_test_function_code --------------------

test_calculate_similarity()