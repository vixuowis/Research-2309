# requirements_file --------------------

!pip install -U torch transformers scikit-learn

# function_import --------------------

import torch
from transformers import BertModel, BertTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def recommend_similar_tv_shows(descriptions):
    """
    Recommends similar TV shows based on their descriptions using sentence embeddings.

    :param descriptions: List of TV show descriptions
    :return: List of tuples with show index and similarity scores
    """
    tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE')
    model = BertModel.from_pretrained('setu4993/LaBSE')
    model = model.eval()

    inputs = tokenizer(descriptions, return_tensors='pt', padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.pooler_output

    similarity_scores = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
            similarity_scores.append(((i, j), similarity))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    return similarity_scores[:10]

# test_function_code --------------------

def test_recommend_similar_tv_shows():
    print("Testing recommend_similar_tv_shows function.")

    # Sample descriptions similar to TV show plot descriptions
    descriptions = [
        'After a plane crash, a doctor and a mysterious man must lead their fellow survivors to safety',
        'A group of survivors on a deserted island fights to stay alive and find rescue',
        'A complex drama about power dynamics in a post-apocalyptic society'
    ]

    # Running the recommendation function
    recommended_shows = recommend_similar_tv_shows(descriptions)

    # Ensure that the function outputs a list of tuples
    assert isinstance(recommended_shows, list), "Output should be a list"
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in recommended_shows), "List elements should be tuples of length 2"

    print("All tests passed.")