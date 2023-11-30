# function_import --------------------

import torch
from transformers import BertModel, BertTokenizerFast

# function_code --------------------

def encode_sentences(sentences):
    """
    Encode sentences using the pre-trained LaBSE model.

    Args:
        sentences (list): A list of sentences to be encoded.

    Returns:
        torch.Tensor: The encoded sentences.
    """
    tokenizer = BertTokenizerFast.from_pretrained('sentence-transformers/LaBSE')
    model = BertModel.from_pretrained('sentence-transformers/LaBSE')

    with torch.no_grad():
        encoded_sentences = []

        for sentence in sentences:
            input_ids, attention_mask = tokenizer(sentence, return_tensors='pt', padding=True)
            
            embeddings = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            encoded_sentences.append(embeddings[:, 0, :])
            
        return torch.vstack(encoded_sentences).squeeze()


# test_function_code --------------------

def test_encode_sentences():
    """
    Test the encode_sentences function.
    """
    sentences = [
        'dog',
        'Cuccioli sono carini.',
        '犬と一緒にビーチを散歩するのが好き',
    ]
    embeddings = encode_sentences(sentences)
    assert embeddings.shape[0] == len(sentences), 'The number of embeddings should be equal to the number of sentences.'
    assert embeddings.shape[1] == 768, 'The dimension of each embedding should be 768.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_encode_sentences()