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
    
    # Check types
    if type(sentences) != list:
        raise TypeError('"sentences" should be of type "list".')
        
    if type(num_clusters) != int:
        raise TypeError('"num_clusters" should be of type "int".')
    
    # Check values
    if num_clusters < 1:
        raise ValueError("Number of clusters must be equal to or greater than 1.")
        
    for sentence in sentences:
        if type(sentence) != str:
            raise TypeError('"sentences" should contain only strings.')
    
    # Create sentence embeddings
    model = SentenceTransformer('stsb-distilbert-base')
    sentence_embeddings = model.encode(sentences, show_progress_bar=True)
        
    # Cluster the embeddings using KMeans
    num_clusters = min(num_clusters, len(sentence_embeddings))
    clustering_model = KMeans(n_clusters=num_clusters)
    
    # cluster_assignment contains the cluster id that sentence with index i is assigned to 
    # (i.e., cluster_assignment[i] is the id of the cluster where sentences[i] has been assigned to)
    cluster_assignment = clustering_model.fit_predict(sentence_embeddings, sample_weight=None)
    
    # Create a dict where each key is an article id and value is a list of sentences with that topic
    groups = {}
    
    for i in range(num_clusters):
        groups[i] = []
        
    for i, sentence in enumerate(sentences):
        groups[cluster_assignment[i]].append(sentence)
            
    return groups

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