# requirements_file --------------------

!pip install -U transformers scikit-learn

# function_import --------------------

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_review_similarity(review1, review2):
    """
    This function calculates the similarity between two book reviews using BERT-based sentence embeddings.

    Args:
        review1 (str): The first book review text.
        review2 (str): The second book review text.

    Returns:
        float: The similarity score between -1 and 1, where higher scores indicate more similarity.
    """
    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    model = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    input_tensors = tokenizer([review1, review2], return_tensors='pt', padding=True, truncation=True)
    embeddings = model(**input_tensors).pooler_output
    similarity_score = cosine_similarity(embeddings[0].detach().numpy().reshape(1, -1), embeddings[1].detach().numpy().reshape(1, -1))[0][0]
    return similarity_score

# test_function_code --------------------

def test_similarity_score():
    print('Testing calculate_review_similarity function.')
    # Two example book reviews
    review1 = 'Absolutely amazing! The characters are unforgettable.'
    review2 = 'Absolutely amazing! The characters are unforgettable.'
    review3 = 'Not enjoyable. Characters are flat and uninteresting.'

    # Test case 1: Identical reviews
    print('Testing identical reviews. Expected score should be close to 1.')
    score = calculate_review_similarity(review1, review2)
    assert 0.99 <= score <= 1, f'Test case failed: Similarity score for identical reviews is {score}'

    # Test case 2: Different reviews
    print('Testing different reviews. Expected score should be significantly lower than 1.')
    score = calculate_review_similarity(review1, review3)
    assert -1 <= score < 0.99, f'Test case failed: Similarity score for different reviews is {score}'

    print('All tests passed.')

# Run the test
if __name__ == '__main__':
    test_similarity_score()