# requirements_file --------------------

!pip install -U sentence-transformers scipy

# function_import --------------------

from sentence_transformers import SentenceTransformer
import scipy.spatial

# function_code --------------------

def find_similar_sentences(sentences, model_name='sentence-transformers/all-MiniLM-L12-v2'):
    # Load the pre-trained SentenceTransformer model
    model = SentenceTransformer(model_name)
    # Generate sentence embeddings
    embeddings = model.encode(sentences)
    # Calculate cosine similarity between sentence embeddings
    cosine_sim_matrix = scipy.spatial.distance.cdist(embeddings, embeddings, 'cosine')
    # Extract the most similar sentences for each sentence based on cosine similarity
    similar_sentences_ids = cosine_sim_matrix.argsort(axis=1)
    return [(sentences[i], [sentences[j] for j in similar_sentences_ids[i][1:] if cosine_sim_matrix[i][j] < 0.5]) for i in range(len(sentences))]

# test_function_code --------------------

def test_find_similar_sentences():
    print("Testing the find_similar_sentences function.")
    # Sample sentences
    sentences = ['This is an example sentence.', 'Each sentence is converted to an embedding.', 'This is another similar sentence.']
    # Expected similar sentences (using a threshold of 0.5 for cosine similarity)
    expected_results = [(sentences[0], [sentences[2]]), (sentences[1], []), (sentences[2], [sentences[0]])]
    # Finding similar sentences
    results = find_similar_sentences(sentences)
    # Test assertion
    assert results == expected_results, f"Failed to find similar sentences: {results}"
    print("Test passed!")

# Run the test
test_find_similar_sentences()