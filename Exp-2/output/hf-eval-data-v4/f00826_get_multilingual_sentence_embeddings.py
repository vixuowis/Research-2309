# requirements_file --------------------

!pip install -U torch transformers

# function_import --------------------

import torch
from transformers import BertModel, BertTokenizerFast

# function_code --------------------

def get_multilingual_sentence_embeddings(sentences):
    """
    Generate embeddings for the provided multilingual sentences using LaBSE model.

    Parameters:
    sentences (list): A list of sentences in various languages.

    Returns:
    embeddings (torch.Tensor): A tensor with the sentence embeddings.
    """
    # Initialize the tokenizer and the model
    tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE')
    model = BertModel.from_pretrained('setu4993/LaBSE')
    model = model.eval()

    # Tokenize the input sentences
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.pooler_output
    return embeddings

# test_function_code --------------------

def test_get_multilingual_sentence_embeddings():
    print("Testing started.")

    # Prepare multilingual data samples
    multilingual_sentences = [
        'How are you?',                # English
        'Come stai?',                  # Italian
        '¿Cómo estás?',                # Spanish
        '你好吗？',                      # Chinese
        'お元気ですか？',               # Japanese
        'Wie geht es Ihnen?',          # German
    ]

    # Get embeddings for the multilingual data samples
    embeddings = get_multilingual_sentence_embeddings(multilingual_sentences)

    # Test cases
    print("Testing embeddings shape.")
    assert embeddings.shape[0] == len(multilingual_sentences), "The number of embeddings should match the number of input sentences."
    assert embeddings.shape[1] == 768, "Embedding size should be 768 for LaBSE."

    print("Testing if embeddings are not None.")
    assert embeddings is not None, "Embeddings should not be None."

    print("Testing finished.")

# Run the test function
test_get_multilingual_sentence_embeddings()