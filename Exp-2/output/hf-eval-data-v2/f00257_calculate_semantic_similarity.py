# function_import --------------------

from transformers import AutoTokenizer, AutoModel

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
    encoded_input1 = tokenizer(text1, padding=True, truncation=True, max_length=128, return_tensors='pt')
    encoded_input2 = tokenizer(text2, padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Generate embeddings for the texts
    embedding1 = model(**encoded_input1)
    embedding2 = model(**encoded_input2)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding1[0], embedding2[0]).item()

    return similarity_score

# test_function_code --------------------

def test_calculate_semantic_similarity():
    """
    Test the calculate_semantic_similarity function.
    """
    text1 = 'The cat sat on the mat.'
    text2 = 'The dog sat on the log.'
    text3 = 'Apple is a tech company.'

    # Similar texts should have a high similarity score
    assert 0.7 <= calculate_semantic_similarity(text1, text2) <= 1.0

    # Dissimilar texts should have a low similarity score
    assert 0.0 <= calculate_semantic_similarity(text1, text3) <= 0.3

# call_test_function_code --------------------

test_calculate_semantic_similarity()