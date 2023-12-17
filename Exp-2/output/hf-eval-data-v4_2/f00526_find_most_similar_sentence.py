# requirements_file --------------------

!pip install -U transformers sklearn torch datasets

# function_import --------------------

from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# function_code --------------------

def find_most_similar_sentence(source_sentence, sentences_to_compare):
    """
    Finds the most similar sentence to the given source sentence from a list.

    Args:
        source_sentence (str): A single source sentence to find similarity against.
        sentences_to_compare (List[str]): A list of sentences to compare with the source sentence.

    Returns:
        Tuple[str, float]: A tuple with the most similar sentence and the similarity score.

    Raises:
        ValueError: If either of the inputs are empty.
    """
    if not source_sentence or not sentences_to_compare:
        raise ValueError("Source sentence or sentences to compare should not be empty.")

    tokenizer = AutoTokenizer.from_pretrained('GanymedeNil/text2vec-large-chinese')
    model = AutoModel.from_pretrained('GanymedeNil/text2vec-large-chinese')

    def encode(sentence):
        input_ids = tokenizer(sentence, return_tensors="pt").input_ids
        return model(input_ids).last_hidden_state.mean(1).detach()

    source_embedding = encode(source_sentence)
    sentence_embeddings = torch.stack([encode(candidate) for candidate in sentences_to_compare])

    similarity_scores = cosine_similarity(source_embedding.cpu(), sentence_embeddings.cpu())
    highest_similarity_index = similarity_scores.argmax()

    most_similar_sentence = sentences_to_compare[highest_similarity_index]
    similarity_score = similarity_scores[0, highest_similarity_index].item()

    return most_similar_sentence, similarity_score

# test_function_code --------------------

from datasets import load_dataset

def test_find_most_similar_sentence():
    print("Testing started.")
    # For testing, we'll use a fictional dataset
    dataset = load_dataset('fictional_chinese_sentences')  # Replace with real dataset if available
    sample_data = dataset['train'][0:3]  # Take first 3 sentences from the dataset

    source_sentence = sample_data[0]
    sentences_to_compare = sample_data[1:]  # Use other two sentences to compare

    expected_sentence = sentences_to_compare[0]  # Assuming the first one is the most similar

    print("Testing case [1/1] started.")
    most_similar_sentence, _ = find_most_similar_sentence(source_sentence, sentences_to_compare)
    assert most_similar_sentence == expected_sentence, f"Test case [1/1] failed: Expected {expected_sentence}, got {most_similar_sentence}"
    print("Testing finished.")

# call_test_function_line --------------------

test_find_most_similar_sentence()