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
    if not type(sentences) == list:
        raise TypeError('The parameter "sentences" must be of type list.')
    elif not type(num_clusters) == int:
        raise TypeError('The parameter "num_cluster" must be an integer.')
    
    embedder = SentenceTransformer("all-MiniLM-L6-v2") # Using model "all-MiniLM-L6-v2".
    sentence_embeddings = embedder.encode(sentences, show_progress_bar=True)
    clusters = KMeans(n_clusters=num_clusters).fit(sentence_embeddings).labels  # Generating cluster labels using KMeans.
    
    # Creating dictionary of cluster ids and a list of the corresponding sentences.
    article_dict = {}
    for i in range(len(clusters)):
        if clusters[i] not in article_dict:
            article_dict[clusters[i]] = [sentences[i]]  # Adding the first sentence to a cluster.
        else:
            article_dict[clusters[i]].append(sentences[i])  # Appending sentences to clusters.
    
    return article_dict


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