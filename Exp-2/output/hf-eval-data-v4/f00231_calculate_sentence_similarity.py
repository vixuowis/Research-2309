# requirements_file --------------------

!pip install -U sentence-transformers scikit-learn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_sentence_similarity(sentence1, sentence2):
    """
    Calculate the similarity score between two sentences using a sentence-transformer model.

    Parameters:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The similarity score between the two sentences (ranges from -1 to 1).
    """
    # Define the sentences
    sentences = [sentence1, sentence2]

    # Create an instance of the SentenceTransformer using the given model
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    # Encode the sentences to get their embeddings
    embeddings = model.encode(sentences)

    # Calculate the cosine similarity between the two embeddings
    similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]

    return similarity

# test_function_code --------------------

def test_calculate_sentence_similarity():
    print("Testing calculate_sentence_similarity function.")

    # Test case 1: Identical sentences
    sentence1 = "I love going to the park"
    sentence2 = "I love going to the park"
    assert calculate_sentence_similarity(sentence1, sentence2) == 1, "Test case 1 failed: Identical sentences should have similarity 1."

    # Test case 2: Completely different sentences
    sentence1 = "I love going to the park"
    sentence2 = "I hate eating vegetables"
    assert calculate_sentence_similarity(sentence1, sentence2) < 0.5, "Test case 2 failed: Completely different sentences should have low similarity."

    # Test case 3: Paraphrased sentences
    sentence1 = "I love going to the park"
    sentence2 = "My favorite activity is visiting the park"
    assert calculate_sentence_similarity(sentence1, sentence2) > 0.5, "Test case 3 failed: Paraphrased sentences should have high similarity."

    print("All test cases passed.")

# Run the test function
test_calculate_sentence_similarity()