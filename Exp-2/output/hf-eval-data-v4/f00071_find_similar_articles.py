# requirements_file --------------------

!pip install -U sentence-transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def find_similar_articles(article_sentence, dataset):
    """
    Find articles in the dataset that are similar to the given article sentence.

    Params:
        article_sentence (str): A sentence representing the main point of the student's research paper.
        dataset (List[str]): A list of sentences from different articles.

    Returns:
        List[tuple]: A list of tuples with each tuple containing the index of the similar article and a similarity score.
    """
    model = SentenceTransformer('sentence-transformers/nli-mpnet-base-v2')
    article_embedding = model.encode(article_sentence)
    dataset_embeddings = model.encode(dataset)

    # Calculate cosine similarities
    similarities = cosine_similarity([article_embedding], dataset_embeddings)[0]
    similar_articles = [(i, score) for i, score in enumerate(similarities) if score > 0.8]

    return similar_articles

# test_function_code --------------------

def test_find_similar_articles():
    print("Testing find_similar_articles function.")
    dataset = [
        "Climate change is leading to an increase in global temperatures.",
        "Effects of global warming on human health.",
        "The effects of climate change on biodiversity in tropical rainforests."
    ]
    research_paper_sentence = "The effects of climate change on biodiversity and ecosystem services in the Arctic."

    similar_articles = find_similar_articles(research_paper_sentence, dataset)
    # Assuming that the last sentence is similar enough
    assert len(similar_articles) > 0, "No similar articles found"
    assert similar_articles[0][0] == 2, "Incorrect similar article index"
    print("All test cases passed.")

# Run the test function
test_find_similar_articles()