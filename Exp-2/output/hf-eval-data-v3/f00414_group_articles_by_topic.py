# function_import --------------------

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans

# function_code --------------------

def group_articles_by_topic(sentences: list, num_clusters: int) -> dict:
    '''
    Groups articles by topic using SentenceTransformer for sentence embeddings and KMeans for clustering.

    Args:
        sentences (list): A list of sentences from the articles.
        num_clusters (int): The number of clusters (topics) to form.

    Returns:
        dict: A dictionary where keys are cluster ids and values are lists of sentences belonging to that cluster.

    Raises:
        ValueError: If sentences is not a list or num_clusters is not an integer.
    '''
    if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
        raise ValueError('sentences must be a list of strings')
    if not isinstance(num_clusters, int):
        raise ValueError('num_clusters must be an integer')

    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    embeddings = model.encode(sentences)
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(embeddings)

    clusters = {i: [] for i in range(num_clusters)}
    for sentence, label in zip(sentences, labels):
        clusters[label].append(sentence)

    return clusters

# test_function_code --------------------

def test_group_articles_by_topic():
    '''Tests the group_articles_by_topic function.'''
    sentences = ['This is an example sentence.', 'Each sentence is converted.', 'This is another example.', 'Each example is different.']
    num_clusters = 2

    clusters = group_articles_by_topic(sentences, num_clusters)

    assert isinstance(clusters, dict), 'Return type must be a dictionary.'
    assert len(clusters) == num_clusters, 'Number of clusters must be equal to num_clusters.'
    for cluster in clusters.values():
        assert all(sentence in sentences for sentence in cluster), 'All sentences in a cluster must be from the input sentences.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_group_articles_by_topic()