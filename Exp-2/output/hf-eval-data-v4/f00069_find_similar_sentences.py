# requirements_file --------------------

!pip install -U sentence-transformers sentencepiece

# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def find_similar_sentences(input_sentence, database_sentences):
    """
    Given an input sentence and a list of sentences from the database, this function finds and returns sentences
    that are semantically similar to the input sentence.

    Args:
    input_sentence (str): The input sentence to compare against the database.
    database_sentences (List[str]): A list of sentences from the database to compare with the input sentence.

    Returns:
    List[str]: A list of sentences from the database that are similar to the input sentence.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v2')
    input_embedding = model.encode([input_sentence], convert_to_tensor=True)
    database_embeddings = model.encode(database_sentences, convert_to_tensor=True)

    # Using cosine similarity to compare the embeddings
    cosine_scores = util.pytorch_cos_sim(input_embedding, database_embeddings)[0]

    # Get the top 5 similar sentences
    top_5_indices = cosine_scores.argsort()[-5:][::-1]
    similar_sentences = [database_sentences[index] for index in top_5_indices]

    return similar_sentences

# test_function_code --------------------

def test_find_similar_sentences():
    print("Testing started.")

    input_sentence = 'This is a sample sentence to test.'
    database_sentences = [
        'This sentence has a similar meaning.',
        'Completely different context and meaning.',
        'Another example sentence for testing purpose.',
        'Semantically similar sentence to the test case.',
        'Random content that is not related.'
    ]

    # Expected similar sentences should have semantic similarity to input_sentence
    expected_similar = [
        'This sentence has a similar meaning.',
        'Semantically similar sentence to the test case.'
    ]

    print("Testing finding similar sentences.")
    similar_sentences = find_similar_sentences(input_sentence, database_sentences)
    assert set(similar_sentences).intersection(set(expected_similar)), "Test case failed: The returned sentences do not match the expected similar sentences."
    print("Test case passed: The returned sentences match the expected similar sentences.")

    print("Testing finished.")

# Run the test function
test_find_similar_sentences()