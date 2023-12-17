# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def recommend_articles(previously_liked, new_article, similarity_threshold=0.8):
    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')
    model = AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')

    # Function description
    """
    Recommend articles to users based on the similarity between the new article
    and their previously liked articles.

    Arguments:
    previously_liked: list of strings, sentences from the articles that user liked.
    new_article: string, the text of the new article to be evaluated.
    similarity_threshold: float, the minimum similarity score required to recommend the article.

    Returns:
    bool: True if the article is recommended, False otherwise.
    """
    # TODO: Implement the function logic using sentence embeddings and similarity comparison
    # Placeholder for actual implementation
    # This should be replaced with logic that computes the embeddings and similarity
    recommended = True

    return recommended

# test_function_code --------------------

def test_recommend_articles():
    print("Testing recommend_articles function.")
    # Example data
    previously_liked = ["I love science articles about space.", "Astronomy news are fascinating."]
    new_article = "The recent discovery in astrophysics opens new possibilities."

    # Test case 1: Similar article
    print("Test case [1/2] started.")
    assert recommend_articles(previously_liked, new_article), "Test case [1/2] failed: the article should be recommended."

    # Test case 2: Dissimilar article
    new_article = "Cooking tips for quick meals."
    print("Test case [2/2] started.")
    assert not recommend_articles(previously_liked, new_article), "Test case [2/2] failed: the article should not be recommended."

    print("Testing finished.")

# Run the test function
test_recommend_articles()