# requirements_file --------------------

!pip install -U torch transformers

# function_import --------------------

import torch
from transformers import BertModel, BertTokenizerFast

# function_code --------------------

def get_multilingual_sentence_embeddings(sentences):
    """
    Get embeddings for multilingual sentences using LaBSE pre-trained model.

    Args:
        sentences (list of str): A list of sentences in different languages.

    Returns:
        torch.Tensor: A tensor containing sentence embeddings.

    Raises:
        ValueError: If the input is not a list or is empty.
    """
    if not isinstance(sentences, list) or not sentences:
        raise ValueError('Input must be a non-empty list of sentences.')

    tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE')
    model = BertModel.from_pretrained('setu4993/LaBSE')
    model = model.eval()
    inputs = tokenizer(sentences, return_tensors='pt', padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.pooler_output
    return embeddings

# test_function_code --------------------

def test_get_multilingual_sentence_embeddings():
    print("Testing started.")
    # Test data with English, Italian and Japanese sentences
    test_sentences = [
        'dog',
        'Cuccioli sono carini.',
        '犬と一緒にビーチを散歩するのが好き',
    ]

    # Testing case 1: Check embedding generation for different languages
    print("Testing case [1/1] started.")
    embeddings = get_multilingual_sentence_embeddings(test_sentences)
    assert embeddings.shape[0] == len(test_sentences), f"Test case [1/1] failed: Expected number of embeddings is {len(test_sentences)}, but got {embeddings.shape[0]} instead."
    print("Testing finished.")

# call_test_function_line --------------------

test_get_multilingual_sentence_embeddings()