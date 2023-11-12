# function_import --------------------

from transformers import AutoTokenizer, AutoModel
import torch

# function_code --------------------

def calculate_semantic_similarity(text1, text2):
    """
    Calculate the semantic similarity between two texts using the 'princeton-nlp/sup-simcse-roberta-large' model.

    Args:
        text1 (str): The first text to compare.
        text2 (str): The second text to compare.

    Returns:
        float: The semantic similarity score between the two texts.
    """
    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-large')
    model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-large')

    # Tokenize and encode the texts
    input1 = tokenizer(text1, return_tensors='pt')
    input2 = tokenizer(text2, return_tensors='pt')

    # Get the embeddings for the texts
    embedding1 = model(**input1)[0]
    embedding2 = model(**input2)[0]

    # Calculate the cosine similarity between the embeddings
    similarity = (embedding1 * embedding2).sum() / (embedding1.norm() * embedding2.norm())

    return similarity.item()

# test_function_code --------------------

def test_calculate_semantic_similarity():
    """
    Test the calculate_semantic_similarity function.
    """
    text1 = 'The cat sat on the mat.'
    text2 = 'The dog sat on the log.'
    text3 = 'Apples are a type of fruit.'

    assert 0.7 <= calculate_semantic_similarity(text1, text2) <= 1.0
    assert 0.0 <= calculate_semantic_similarity(text1, text3) <= 0.3
    assert 0.0 <= calculate_semantic_similarity(text2, text3) <= 0.3

    return 'All Tests Passed'

# call_test_function_code --------------------

test_calculate_semantic_similarity()