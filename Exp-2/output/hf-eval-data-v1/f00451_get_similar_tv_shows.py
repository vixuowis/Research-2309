import torch
from transformers import BertModel, BertTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity

def get_similar_tv_shows(tv_show_descriptions):
    '''
    This function takes a list of TV show descriptions and returns a similarity matrix.
    The similarity is calculated using a BERT-based model trained on sentence embedding.
    '''
    tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE')
    model = BertModel.from_pretrained('setu4993/LaBSE')
    model = model.eval()

    inputs = tokenizer(
      tv_show_descriptions,
      return_tensors='pt',
      padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.pooler_output

    similarity_matrix = cosine_similarity(embeddings)

    return similarity_matrix