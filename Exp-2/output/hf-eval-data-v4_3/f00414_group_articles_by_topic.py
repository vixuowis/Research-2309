# requirements_file --------------------

import subprocess

requirements = ["sentence-transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def group_articles_by_topic(articles, threshold=0.75):
    """
    Groups articles into clusters based on their semantic similarity.

    Args:
        articles (list): List of articles to be grouped.
        threshold (float): Cosine similarity threshold for clustering articles.

    Returns:
        dict: A dictionary where keys are cluster IDs and values are lists of article indices belonging to that cluster.

    Raises:
        ValueError: If articles is not a list or threshold is not a valid float value.
    """
    # Check input validity
    if not isinstance(articles, list) or not all(isinstance(article, str) for article in articles):
        raise ValueError('The articles must be a list of strings.')
    if not isinstance(threshold, float) or not (0 <= threshold <= 1):
        raise ValueError('Threshold must be a float between 0 and 1.')

    # Initialize model and encode articles
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    embeddings = model.encode(articles)

    # Compute pairwise cosine similarity and cluster articles
    clusters = {}
    for i, emb_i in enumerate(embeddings):
        for j, emb_j in enumerate(embeddings):
            if i != j and cosine_similarity(emb_i, emb_j) > threshold:
                clusters.setdefault(i, []).append(j)

    return clusters

# test_function_code --------------------

def test_group_articles_by_topic():
    print("Testing started.")

    # Assuming load_dataset is a custom function that loads test data
    articles = [
        'This is an article about artificial intelligence.',
        'Este es un artículo sobre inteligencia artificial.',
        'Dies ist ein Artikel über künstliche Intelligenz.',
        'Another distinct topic for control.'
    ]

    # Test case 1: Similar articles should be grouped
    print("Testing case [1/2] started.")
    clusters = group_articles_by_topic(articles, threshold=0.75)
    assert len(clusters) >= 1, f"Test case [1/2] failed: Expected at least one cluster, got {len(clusters)} clusters."

    # Test case 2: Exception handling
    print("Testing case [2/2] started.")
    try:
        group_articles_by_topic('not a list')
        assert False, "Test case [2/2] failed: ValueError expected."
    except ValueError as e:
        pass

    print("Testing finished.")

# Call the test function
test_group_articles_by_topic()

# call_test_function_line --------------------

test_group_articles_by_topic()