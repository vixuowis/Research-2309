# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def compute_semantic_similarity(text1, text2):
    """
    Compute the semantic similarity between two texts using the sup-simcse-roberta-large model.

    Args:
        text1 (str): The first text to compare.
        text2 (str): The second text to compare.

    Returns:
        float: A semantic similarity score between the two texts.

    Raises:
        ValueError: If any of the text inputs is None or empty.

    """
    if not text1 or not text2:
        raise ValueError('Both text inputs must be non-empty strings.')
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-large')
    model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-large')
    
    # Tokenize texts
    tokens1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True)
    tokens2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True)

    # Get sentence embeddings
    with torch.no_grad():
        embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1)

    # Calculate cosine similarity as semantic similarity
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)
    score = cosine_similarity(embeddings1, embeddings2).item()

    return score

# test_function_code --------------------

def test_compute_semantic_similarity():
    print("Testing started.")

    # Sample texts
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "A fast dark-colored fox leaps above a sleepy canine."
    text3 = ""

    # Testing case 1: Non-empty strings
    print("Testing case [1/3] started.")
    similarity_score = compute_semantic_similarity(text1, text2)
    assert similarity_score >= 0 and similarity_score <= 1, f"Test case [1/3] failed: Similarity score out of bounds {similarity_score}"

    # Testing case 2: Same text
    print("Testing case [2/3] started.")
    similarity_score_same = compute_semantic_similarity(text1, text1)
    assert similarity_score_same == 1, f"Test case [2/3] failed: Similarity score should be 1 for identical texts, got {similarity_score_same}"

    # Testing case 3: Empty string
    print("Testing case [3/3] started.")
    try:
        compute_semantic_similarity(text1, text3)
        assert False, "Test case [3/3] failed: ValueError was not raised for empty string input."
    except ValueError as e:
        assert str(e) == 'Both text inputs must be non-empty strings.', f"Test case [3/3] failed: Incorrect error message {str(e)}"

    print("Testing finished.")

# call_test_function_line --------------------

test_compute_semantic_similarity()