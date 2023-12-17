# requirements_file --------------------

!pip install -U sentence-transformers 

# function_import --------------------

from sentence_transformers import SentenceTransformer, util

# function_code --------------------

def find_most_relevant_sentence(question, sentences):
    """
    Find the most relevant sentence to the given question using a pre-trained SentenceTransformer model.

    Parameters:
    question (str): A question string.
    sentences (List[str]): A list of sentences among which to find the relevant one.

    Returns:
    str: The most relevant sentence to the question.
    """
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    question_emb = model.encode(question)
    sentences_emb = model.encode(sentences)
    scores = util.dot_score(question_emb, sentences_emb)[0].cpu().tolist()
    best_sentence_index = scores.index(max(scores))
    return sentences[best_sentence_index]

# test_function_code --------------------

def test_find_most_relevant_sentence():
    print("Testing find_most_relevant_sentence function.")
    question = "What is the main purpose of photosynthesis?"
    sentences = [
        "Photosynthesis is the process used by plants, algae, and some bacteria to harness energy from sunlight into chemical energy.",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "The process of photosynthesis converts light energy to chemical energy, which can be used by organisms for various metabolic processes."]

    expected = sentences[0]
    result = find_most_relevant_sentence(question, sentences)

    assert result == expected, f"Test failed: Expected '{{expected}}', but got '{{result}}'"
    print("Test passed.")

# Run the test
test_find_most_relevant_sentence()