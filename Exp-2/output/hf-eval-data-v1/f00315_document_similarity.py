from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def document_similarity(documents):
    '''
    This function takes a list of documents as input and returns a similarity matrix.
    The similarity is calculated using the SentenceTransformer model from the Hugging Face Transformers library.
    The model used is 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'.
    Each document is converted into a 384-dimensional vector (embedding), and the cosine similarity between these embeddings is calculated.
    '''
    # Initialize the SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Convert each document into a dense vector (embedding) in a 384-dimensional space
    embeddings = model.encode(documents)
    
    # Compute the cosine similarity between the embeddings
    similarity_matrix = cosine_similarity(embeddings)
    
    return similarity_matrix