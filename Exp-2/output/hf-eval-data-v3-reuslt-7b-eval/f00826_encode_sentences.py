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
    # Tokenize with the LaBSE tokenizer.
    tokenized = [tokenizer(sentence)["input_ids"] for sentence in sentences]
    
    # Pad all the sequences to the same length.
    lengths = [len(sequence) for sequence in tokenized]
    padded   = torch.zeros((len(tokenized), max(lengths)), dtype=torch.long)

    # Replace with the actual values.
    for index, (seq, length) in enumerate(zip(tokenized, lengths)):
        padded[index, :length] = torch.tensor(seq[:length])
    
    # Encode all of these sentences together using the LaBSE model.
    with torch.no_grad():
        encoded = model(padded)['sentence_embedding']

    return encoded

# main --------------------

if __name__ == "__main__":
    # Initialize the tokenizer and pre-trained LaBSE model.
    model      = BertModel.from_pretrained("monologg/LaBSE")
    tokenizer  = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Encode a list of sentences.
    sentences = [
        "I love the smell of burnt rubber in the morning.",
        "You're so ugly I can't see anything else",
        "I am not afraid of any monster under my bed."
    ]
    embedded  = encode_sentences(sentences)
    print(embedded.shape)

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