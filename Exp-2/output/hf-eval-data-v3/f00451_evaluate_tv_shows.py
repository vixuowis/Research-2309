# function_import --------------------

import torch
from transformers import BertModel, BertTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def evaluate_tv_shows(tv_show_descriptions):
    """
    Evaluate TV shows using a BERT-based model trained on sentence embedding to find the most similar TV shows based on description.

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

def test_evaluate_tv_shows():
    """
    Test the function evaluate_tv_shows.
    """
    tv_show_descriptions = [
      'A group of friends navigate through their complicated lives.',
      'An alien stranded on Earth tries to fit in.',
      'A detective solves mysteries with his unique perspective.'
    ]
    similarity_matrix = evaluate_tv_shows(tv_show_descriptions)
    assert similarity_matrix.shape == (3, 3), 'The shape of the similarity matrix is incorrect.'
    assert similarity_matrix.dtype == np.float32, 'The data type of the similarity matrix is incorrect.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_evaluate_tv_shows()