# requirements_file --------------------

!pip install -U transformers sklearn

# function_import --------------------

from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# function_code --------------------

def find_most_similar_chinese_sentence(source_sentence, sentences_to_compare):
    tokenizer = AutoTokenizer.from_pretrained('GanymedeNil/text2vec-large-chinese')
    model = AutoModel.from_pretrained('GanymedeNil/text2vec-large-chinese')

    def encode(sentence):
        input_ids = tokenizer(sentence, return_tensors="pt").input_ids
        embeddings = model(input_ids)
        return embeddings.last_hidden_state.mean(1).detach()

    source_embedding = encode(source_sentence)
    sentence_embeddings = torch.stack([encode(candidate) for candidate in sentences_to_compare])

    similarity_scores = cosine_similarity(source_embedding.cpu(), sentence_embeddings.cpu())
    highest_similarity_index = similarity_scores.argmax()

    return sentences_to_compare[highest_similarity_index], similarity_scores[0][highest_similarity_index].item()

# test_function_code --------------------

def test_find_most_similar_chinese_sentence():
    source = '这是一个源句子'
    candidates = ['这是第一个候选句子', '这是一个源句子', '这是第二个候选句子']
    expected_sentence = '这是一个源句子'
    most_similar_sentence, similarity_score = find_most_similar_chinese_sentence(source, candidates)

    assert most_similar_sentence == expected_sentence, f"Test failed: Expected {expected_sentence}, but got {most_similar_sentence}"