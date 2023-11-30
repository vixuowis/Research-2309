# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_sentence_embeddings(input_text: str):
    '''
    This function takes a sentence as input and returns its embedding using the LaBSE model.
    
    Args:
    input_text (str): The sentence to be encoded.
    
    Returns:
    Tensor: The sentence embedding.
    '''

    # Initializing labse tokenizer and model
    model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    
    input_ids = tokenizer(input_text, padding="max_length", max_length=512, return_tensors='pt').input_ids
    embeddings = model(input_ids)[0]  # Batch size 1
    
    return embeddings

# test_function_code --------------------

def test_extract_sentence_embeddings():
    '''
    This function tests the extract_sentence_embeddings function.
    '''
    sentence1 = 'Here is a sentence in English.'
    sentence2 = 'Voici une phrase en français.'
    sentence3 = 'Aquí hay una frase en español.'
    
    embedding1 = extract_sentence_embeddings(sentence1)
    embedding2 = extract_sentence_embeddings(sentence2)
    embedding3 = extract_sentence_embeddings(sentence3)
    
    assert embedding1.shape == (1, 768)
    assert embedding2.shape == (1, 768)
    assert embedding3.shape == (1, 768)
    
    return 'All Tests Passed'


# call_test_function_code --------------------

test_extract_sentence_embeddings()