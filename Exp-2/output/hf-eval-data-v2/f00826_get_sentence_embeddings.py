# function_import --------------------

import torch
from transformers import BertModel, BertTokenizerFast

# function_code --------------------

def get_sentence_embeddings(sentences):
    """
    This function takes a list of sentences in various languages and returns their embeddings using the LaBSE model.

    Args:
        sentences (list): A list of sentences. Each sentence is a string.

    Returns:
        embeddings (torch.Tensor): A tensor containing the embeddings of the input sentences.
    """
    tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE')
    model = BertModel.from_pretrained('setu4993/LaBSE')
    model = model.eval()
    inputs = tokenizer(sentences, return_tensors='pt', padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.pooler_output
    return embeddings

# test_function_code --------------------

def test_get_sentence_embeddings():
    """
    This function tests the get_sentence_embeddings function by comparing the output embeddings for a set of test sentences.
    """
    test_sentences = [
        'dog',
        'Cuccioli sono carini.',
        '犬と一緒にビーチを散歩するのが好き',
    ]
    embeddings = get_sentence_embeddings(test_sentences)
    assert embeddings.shape[0] == len(test_sentences), 'The number of embeddings should be equal to the number of input sentences.'
    assert embeddings.shape[1] == 768, 'The dimension of each embedding should be 768.'

# call_test_function_code --------------------

test_get_sentence_embeddings()