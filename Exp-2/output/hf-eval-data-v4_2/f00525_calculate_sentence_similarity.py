# requirements_file --------------------

!pip install -U sentence-transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_sentence_similarity(sentences):
    """
    Generates similarity scores between sentences using a pre-trained model.

    Args:
        sentences (List[str]): A list of sentences to be compared for similarity.

    Returns:
        numpy.ndarray: A 2D array of similarity scores between all pairs of sentences.

    Raises:
        ValueError: If the list of sentences is empty.

    """
    if not sentences:
        raise ValueError('The sentence list must not be empty.')

    # Load the pre-trained model
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Encode the sentences to get their embeddings
    embeddings = model.encode(sentences)

    # Calculate the cosine similarity between sentence embeddings
    similarity_scores = cosine_similarity(embeddings)

    return similarity_scores

# test_function_code --------------------

def test_calculate_sentence_similarity():
    print('Testing started.')
    sentences = [
        'This is an example sentence.',
        'Each sentence is converted.',
        'Calculate the similarity between sentences.'
    ]

    # Test case 1: Correct similarity computation
    print('Testing case [1/1] started.')
    similarity_scores = calculate_sentence_similarity(sentences)
    assert similarity_scores.shape == (len(sentences), len(sentences)), f'Test case [1/1] failed: Shape mismatch.'
    print('Testing finished.')

# call_test_function_line --------------------

test_calculate_sentence_similarity()