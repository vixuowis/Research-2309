from sentence_transformers import SentenceTransformer

def encode_sentences(sentences):
    '''
    This function takes a list of sentences as input and returns their embeddings.
    The embeddings are generated using the pre-trained model 'sentence-transformers/all-distilroberta-v1'.
    These embeddings can be used for tasks like clustering, similarity analysis or semantic search.
    '''
    # Instantiate a SentenceTransformer model using the pre-trained model 'sentence-transformers/all-distilroberta-v1'
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    # Encode the sentences into a 768-dimensional dense vector space
    embeddings = model.encode(sentences)
    return embeddings