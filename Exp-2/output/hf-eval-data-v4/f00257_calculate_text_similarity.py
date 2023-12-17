# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModel
import torch

# function_code --------------------

def calculate_text_similarity(text1, text2):
    """
    Calculate the semantic similarity between two texts using the sup-simcse-roberta-large model.

    Parameters:
        text1 (str): The first text to compare.
        text2 (str): The second text to compare.

    Returns:
        float: The semantic similarity score between the two texts.
    """
    # Load the tokenizer and model from the pretrained sup-simcse-roberta-large
    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-large')
    model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-large')

    # Tokenize and encode the text inputs
    inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True)

    # Obtain the embeddings for both texts
    with torch.no_grad():
        embeddings1 = model(**inputs1)[0]
        embeddings2 = model(**inputs2)[0]

    # Calculate the cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
    return cosine_sim.item()

# test_function_code --------------------

def test_calculate_text_similarity():
    print("Testing similarity calculation.")

    # Test case 1: Similar texts
    assert calculate_text_similarity('The quick brown fox jumps over the lazy dog', 'A swift auburn fox leaps over the inactive canine') > 0.7, "Test case failed: Similar texts should have high similarity score"

    # Test case 2: Dissimilar texts
    assert calculate_text_similarity('The quick brown fox jumps over the lazy dog', 'Lorem ipsum dolor sit amet, consectetur adipiscing elit') < 0.3, "Test case failed: Dissimilar texts should have low similarity score"

    # Test case 3: Identical texts
    score = calculate_text_similarity('The quick brown fox jumps over the lazy dog', 'The quick brown fox jumps over the lazy dog')
    assert score == 1.0, "Test case failed: Identical texts should have the highest similarity score"
    print("Testing finished.")

test_calculate_text_similarity()