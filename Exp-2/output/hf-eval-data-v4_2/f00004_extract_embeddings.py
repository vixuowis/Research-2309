# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_embeddings(input_text):
    """
    Extracts the embeddings for the input text using the LaBSE model.

    Args:
        input_text (str): The text for which the embeddings will be extracted.

    Returns:
        torch.Tensor: The embeddings of the input text.

    Raises:
        ValueError: If the input_text is not a string.
    """
    if not isinstance(input_text, str):
        raise ValueError('Input text must be a string.')

    model = AutoModel.from_pretrained('rasa/LaBSE')
    tokenizer = AutoTokenizer.from_pretrained('rasa/LaBSE')
    encoded_input = tokenizer(input_text, return_tensors='pt')
    embeddings = model(**encoded_input)
    return embeddings.pooler_output

# test_function_code --------------------

def test_extract_embeddings():
    print("Testing started.")

    # Testing case 1: Extracting embeddings for English text
    print("Testing case [1/2] started.")
    english_text = "This is a test sentence in English."
    embeddings_en = extract_embeddings(english_text)
    assert embeddings_en is not None and embeddings_en.shape[0] == 1, f"Test case [1/2] failed: Expected embeddings shape to be (1, embedding_dim)."

    # Testing case 2: Testing for invalid input type
    print("Testing case [2/2] started.")
    try:
        extract_embeddings(123)  # Invalid input type
        assert False, "Test case [2/2] failed: ValueError expected for non-string input."
    except ValueError as e:
        assert str(e) == 'Input text must be a string.', "Test case [2/2] failed: {str(e)}"

    print("Testing finished.")

# call_test_function_line --------------------

test_extract_embeddings()