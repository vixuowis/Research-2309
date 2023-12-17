# requirements_file --------------------

!pip install -U sentence-transformers transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def find_similar_lyrics(sentences, similarity_threshold=0.8):
    '''
    This function takes a list of sentences (lyrics) and returns pairs of sentences that are similar to each other based on a specified similarity threshold.

    Args:
        sentences (list of str): The list of sentences to compare.
        similarity_threshold (float, optional): The threshold for considering sentences as similar. Defaults to 0.8.

    Returns:
        list of tuples: A list containing tuples of similar sentence indices.
    '''
    # Load the pre-trained sentence-transformers model
    model = SentenceTransformer('nikcheerla/nooks-amd-detection-v2-full')

    # Encode the sentences to get the embeddings
    embeddings = model.encode(sentences)

    # Compute pairwise cosine similarity
    cos_sim_matrix = cosine_similarity(embeddings)

    # Find pairs of sentences that have similarity above the threshold
    similar_pairs = []
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            if cos_sim_matrix[i][j] > similarity_threshold:
                similar_pairs.append((i, j))

    return similar_pairs

# test_function_code --------------------

def test_find_similar_lyrics():
    print("Testing the find_similar_lyrics function.")

    # Sample sentences with similarity
    sentences = [
        'I love the way the sun sets at the beach',
        'The sun sinking into the ocean is a beautiful sight',
        'I enjoy walking my dog in the park',
    ]

    # Expected to find one pair of similar sentences
    expected = [(0, 1)]

    # Obtain the actual similar pairs
    actual = find_similar_lyrics(sentences)

    # Testing if the actual similar pairs match the expected ones
    assert actual == expected, f"Test failed: Expected {expected}, but got {actual}."

    print("Test passed successfully.")

test_find_similar_lyrics()