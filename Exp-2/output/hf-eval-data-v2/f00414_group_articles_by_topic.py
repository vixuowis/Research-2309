# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def group_articles_by_topic(sentences):
    '''
    Group articles discussing the specific topic using SentenceTransformer.
    
    Args:
        sentences (list): A list of sentences from different articles.
    
    Returns:
        embeddings (list): A list of embeddings for each sentence.
    
    '''
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    embeddings = model.encode(sentences)
    return embeddings

# test_function_code --------------------

def test_group_articles_by_topic():
    '''
    Test the function group_articles_by_topic.
    
    '''
    sentences = ['This is an example sentence', 'Each sentence is converted']
    embeddings = group_articles_by_topic(sentences)
    assert len(embeddings) == len(sentences), 'The number of embeddings should be equal to the number of sentences.'
    assert len(embeddings[0]) == 512, 'The dimension of each embedding should be 512.'

# call_test_function_code --------------------

test_group_articles_by_topic()