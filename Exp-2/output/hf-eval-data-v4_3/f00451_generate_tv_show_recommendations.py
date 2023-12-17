# requirements_file --------------------

import subprocess

requirements = ["torch", "transformers", "scikit-learn"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import torch
from transformers import BertModel, BertTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity


# function_code --------------------

def generate_tv_show_recommendations(descriptions):
    """
    Generate recommendations for TV shows based on the descriptions
    using BERT-based model sentence embeddings.

    Args:
        descriptions (list of str): Description of TV shows.

    Returns:
        list of (int, int, float): List of tuples containing the indices of the two
        recommended TV shows and their similarity score.

    Raises:
        ValueError: If descriptions is empty.

    """
    if not descriptions:
        raise ValueError('The descriptions list is empty.')

    # Initialize tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE')
    model = BertModel.from_pretrained('setu4993/LaBSE')
    model = model.eval()

    # Tokenize descriptions
    inputs = tokenizer(descriptions, return_tensors='pt', padding=True, truncation=True)

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.pooler_output

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Recommendation logic: find most similar pairs
    recommendations = []
    for i in range(len(descriptions)):
        for j in range(i + 1, len(descriptions)):
            score = similarity_matrix[i][j]
            recommendations.append((i, j, score))

    # Sort by highest scores
    recommendations.sort(key=lambda x: x[2], reverse=True)

    return recommendations


# test_function_code --------------------

def test_generate_tv_show_recommendations():
    print("Testing started.")
    sample_descriptions = [
        'Two noble families fight for control of the mythical land of Westeros.',
        'A group of survivors are forced to work together after a zombie apocalypse.',
        'A modern update finds the famous sleuth and his doctor partner solving crime in 21st century London.'
    ]

    # Test case 1
    print("Testing case [1/1] started.")
    recommendations = generate_tv_show_recommendations(sample_descriptions)
    assert recommendations, f"Test case [1/1] failed: No recommendations generated."
    assert len(recommendations) == 3, f"Test case [1/1] failed: Expected 3 recommendations, got {len(recommendations)}"
    assert all(isinstance(rec, tuple) for rec in recommendations), f"Test case [1/1] failed: Recommendations should be tuples."
    assert all(len(rec) == 3 for rec in recommendations), f"Test case [1/1] failed: Each recommendation should be a tuple of length 3."
    print("Testing finished.")


# call_test_function_line --------------------

test_generate_tv_show_recommendations()