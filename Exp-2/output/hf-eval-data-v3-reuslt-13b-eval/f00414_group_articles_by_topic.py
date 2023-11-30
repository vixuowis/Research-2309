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
    # Checking the type of the argument 'sentences'
    if(type(sentences) != list):
        raise TypeError("'sentences' should be of type 'list'")
    
    # Checking the type of the argument 'num_clusters'
    if(type(num_clusters) != int):
        raise TypeError("'num_clusters' should be of type 'int'")

    model = SentenceTransformer('all-mpnet-base-v2')
    sentence_embeddings = model.encode(sentences, show_progress_bar=True)
    
    kmeans_model = KMeans(n_clusters=num_clusters).fit(np.array(sentence_embeddings))
    labels = kmeans_model.labels_

    result_dict = {}
    for idx, cluster in enumerate(labels):
        if str(cluster) not in result_dict:
            result_dict[str(cluster)] = []
        
        result_dict[str(cluster)].append((idx, sentences[idx]))
    
    return result_dict

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