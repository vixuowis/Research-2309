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
    # load model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    model = AutoModel.from_pretrained('sentence-transformers/LaBSE')
    
    encoded_input = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    # sentence embeddings
    embedding = model_output.pooler_output[0]
    
    return embedding

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