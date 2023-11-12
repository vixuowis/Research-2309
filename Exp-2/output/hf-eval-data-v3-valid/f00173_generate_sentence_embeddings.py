# function_import --------------------

from transformers import AutoTokenizer, AutoModel
import torch

# function_code --------------------

def generate_sentence_embeddings(sentences):
    """
    Generate sentence embeddings for the given sentences using a pre-trained model.

    Args:
        sentences (list): A list of sentences for which to generate embeddings.

    Returns:
        torch.Tensor: A tensor containing the sentence embeddings.
    """
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/sbert_large_mt_nlu_ru')
    model = AutoModel.from_pretrained('sberbank-ai/sbert_large_mt_nlu_ru')

    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

# test_function_code --------------------

def test_generate_sentence_embeddings():
    """
    Test the generate_sentence_embeddings function.
    """
    sentences = ['Анализировать текст российской газеты', 'Это просто пример предложения', 'Мы тестируем функцию генерации вложений предложений']
    embeddings = generate_sentence_embeddings(sentences)
    assert embeddings.shape[0] == len(sentences), 'Number of embeddings does not match number of sentences'
    assert embeddings.shape[1] == 1024, 'Embedding size does not match expected size'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_sentence_embeddings()