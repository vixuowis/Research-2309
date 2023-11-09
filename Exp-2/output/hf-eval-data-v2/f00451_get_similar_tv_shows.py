# function_import --------------------

import torch
from transformers import BertModel, BertTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def get_similar_tv_shows(tv_show_descriptions):
    """
    This function evaluates TV shows using a BERT-based model trained on sentence embedding to find the most similar TV shows based on description.

    Args:
        tv_show_descriptions (list): A list of descriptions of the TV shows.

    Returns:
        similarity_matrix (numpy.ndarray): A matrix of similarity scores between TV shows.
    """
    tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE')
    model = BertModel.from_pretrained('setu4993/LaBSE')
    model = model.eval()

    inputs = tokenizer(
      tv_show_descriptions,
      return_tensors='pt',
      padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.pooler_output

    similarity_matrix = cosine_similarity(embeddings)

    return similarity_matrix

# test_function_code --------------------

def test_get_similar_tv_shows():
    """
    This function tests the get_similar_tv_shows function.
    """
    tv_show_descriptions = [
      'A group of friends navigate through life in New York City.',
      'A group of scientists work in a secret government lab.',
      'A group of survivors navigate through a post-apocalyptic world.'
    ]
    similarity_matrix = get_similar_tv_shows(tv_show_descriptions)
    assert similarity_matrix.shape == (len(tv_show_descriptions), len(tv_show_descriptions))

# call_test_function_code --------------------

test_get_similar_tv_shows()