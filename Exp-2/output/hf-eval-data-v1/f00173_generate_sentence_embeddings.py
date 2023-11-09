from transformers import AutoTokenizer, AutoModel
import torch


def mean_pooling(model_output, attention_mask):
    """
    This function performs mean pooling on the model output.
    It takes into account the attention mask for correct averaging.
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def generate_sentence_embeddings(sentences):
    """
    This function generates sentence embeddings for the given sentences.
    It uses the pre-trained 'sberbank-ai/sbert_large_mt_nlu_ru' model from Hugging Face Transformers.
    """
    tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/sbert_large_mt_nlu_ru')
    model = AutoModel.from_pretrained('sberbank-ai/sbert_large_mt_nlu_ru')

    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings