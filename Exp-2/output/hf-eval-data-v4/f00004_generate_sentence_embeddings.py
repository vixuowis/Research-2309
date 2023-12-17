# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def generate_sentence_embeddings(text, model_name='rasa/LaBSE'):
    """
    Generate embeddings for a given sentence using a pre-trained LaBSE model.

    Args:
        text (str): The input sentence text.
        model_name (str): The pre-trained model to use. Defaults to 'rasa/LaBSE'.

    Returns:
        torch.Tensor: The sentence embedding.
    """
    # Instantiate the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Encode the text
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Generate the embeddings
    with torch.no_grad():  # Disable gradient calculations
        model_output = model(**encoded_input)

    # Retrieve the pooler output (sentence embedding)
    sentence_embedding = model_output.pooler_output
    return sentence_embedding

# test_function_code --------------------

def test_generate_sentence_embeddings():
    print("Testing generate_sentence_embeddings started.")
    input_text = 'Here is a sentence in English.'

    try:
        # This should succeed
        embedding = generate_sentence_embeddings(input_text)
        assert embedding is not None, f"Test failed: Embedding is None"
    except Exception as e:
        assert False, f"Test failed with exception: {e}"

    print("Testing generate_sentence_embeddings finished.")

# Run the test
test_generate_sentence_embeddings()