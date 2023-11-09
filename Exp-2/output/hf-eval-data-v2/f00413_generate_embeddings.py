# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def generate_embeddings(sentences):
    """
    This function generates sentence embeddings using the SentenceTransformer model.

    Args:
        sentences (list): A list of sentences for which to generate embeddings.

    Returns:
        embeddings (list): A list of embeddings for the input sentences.
    """
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    embeddings = model.encode(sentences)
    return embeddings

# test_function_code --------------------

def test_generate_embeddings():
    """
    This function tests the generate_embeddings function.
    It uses a sample sentence and checks if the output is a list of embeddings.
    """
    sentences = ['This is an example sentence', 'Each sentence is converted']
    embeddings = generate_embeddings(sentences)
    assert isinstance(embeddings, list), 'The output should be a list of embeddings.'
    assert len(embeddings) == len(sentences), 'The number of embeddings should be equal to the number of sentences.'

# call_test_function_code --------------------

test_generate_embeddings()