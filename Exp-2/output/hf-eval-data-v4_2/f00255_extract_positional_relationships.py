# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def extract_positional_relationships(text):
    """
    Extracts the [CLS] embedding from the last layer of the pretrained SapBERT model representing positional relations in medical texts.

    Args:
        text (str): A string containing the biomedical entities.

    Returns:
        torch.Tensor: Embedding vector indicating the position of embedded biomedical entities.

    Raises:
        ValueError: If the text is empty.
    """
    if not text:
        raise ValueError('Input text cannot be empty.')
    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# test_function_code --------------------

def test_extract_positional_relationships():
    print("Testing started.")
    test_cases = [
        "covid infection",
        "",
        "lung cancer treatment"
    ]

    # Test case 1: Valid medical text
    print("Testing case [1/3] started.")
    cls_emb_1 = extract_positional_relationships(test_cases[0])
    assert cls_emb_1 is not None, f"Test case [1/3] failed: Expected a valid embedding, got None"

    # Test case 2: Empty text
    print("Testing case [2/3] started.")
    try:
        extract_positional_relationships(test_cases[1])
        assert False, "Test case [2/3] failed: Expected a ValueError for empty text"
    except ValueError:
        pass  # Expected exception

    # Test case 3: Another valid medical text
    print("Testing case [3/3] started.")
    cls_emb_3 = extract_positional_relationships(test_cases[2])
    assert cls_emb_3 is not None, f"Test case [3/3] failed: Expected a valid embedding, got None"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_positional_relationships()