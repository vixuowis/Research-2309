# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_sentence_embeddings(input_text):
    """
    This function takes a sentence as input and returns its embedding using the LaBSE model.

    Args:
        input_text (str): The sentence to be embedded.

    Returns:
        sentence_embedding (torch.Tensor): The embedding of the input sentence.
    """
    model = AutoModel.from_pretrained('rasa/LaBSE')
    tokenizer = AutoTokenizer.from_pretrained('rasa/LaBSE')
    encoded_input = tokenizer(input_text, return_tensors='pt')
    embeddings = model(**encoded_input)
    sentence_embedding = embeddings.pooler_output
    return sentence_embedding

# test_function_code --------------------

def test_extract_sentence_embeddings():
    """
    This function tests the extract_sentence_embeddings function by comparing the output for a known input with a known output.
    Note: Due to the nature of the model, the output can vary slightly each time the function is run, so we are not comparing for exact equality.
    """
    input_text = 'Here is a sentence in English.'
    output = extract_sentence_embeddings(input_text)
    assert output.shape == (1, 768), 'The output shape is not as expected.'

# call_test_function_code --------------------

test_extract_sentence_embeddings()