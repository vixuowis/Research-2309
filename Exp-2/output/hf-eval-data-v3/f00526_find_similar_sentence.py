# function_import --------------------

from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# function_code --------------------

def find_similar_sentence(source_sentence: str, sentences_to_compare: list) -> str:
    """
    Find the most similar sentence to the source sentence from a list of sentences.

    Args:
        source_sentence (str): The source sentence to compare.
        sentences_to_compare (list): The list of sentences to compare with the source sentence.

    Returns:
        str: The most similar sentence from the list.

    Raises:
        OSError: If there is not enough disk space to download the model.
    """
    tokenizer = AutoTokenizer.from_pretrained('GanymedeNil/text2vec-large-chinese')
    model = AutoModel.from_pretrained('GanymedeNil/text2vec-large-chinese')

    def encode(sentence):
        input_ids = tokenizer(sentence, return_tensors='pt').input_ids
        return model(input_ids).last_hidden_state.mean(1).detach()

    source_embedding = encode(source_sentence)
    sentence_embeddings = torch.stack([encode(candidate) for candidate in sentences_to_compare])

    similarity_scores = cosine_similarity(source_embedding.cpu(), sentence_embeddings.cpu())
    highest_similarity_index = similarity_scores.argmax()

    return sentences_to_compare[highest_similarity_index]

# test_function_code --------------------

def test_find_similar_sentence():
    source_sentence = '我爱吃苹果'
    sentences_to_compare = ['我喜欢吃香蕉', '我爱吃苹果', '我不喜欢吃橙子']
    assert find_similar_sentence(source_sentence, sentences_to_compare) == '我爱吃苹果'

    source_sentence = '我喜欢跑步'
    sentences_to_compare = ['我喜欢游泳', '我喜欢跑步', '我喜欢打篮球']
    assert find_similar_sentence(source_sentence, sentences_to_compare) == '我喜欢跑步'

    source_sentence = '我喜欢看电影'
    sentences_to_compare = ['我喜欢看书', '我喜欢看电影', '我喜欢听音乐']
    assert find_similar_sentence(source_sentence, sentences_to_compare) == '我喜欢看电影'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_find_similar_sentence()