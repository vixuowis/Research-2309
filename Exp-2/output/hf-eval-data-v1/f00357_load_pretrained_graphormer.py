from transformers import AutoModel


def load_pretrained_graphormer():
    """
    This function loads a pretrained Graphormer model from Hugging Face Transformers.
    The model is 'graphormer-base-pcqm4mv1', a graph Transformer model pretrained on the PCQM4M-LSC dataset.
    This dataset has been used to perform quantum property predictions on molecular graphs, which makes it suitable for molecular property prediction tasks.
    The pretrained Graphormer model has achieved 1st place in the KDD CUP 2021 for quantum prediction.
    """
    model = AutoModel.from_pretrained('graphormer-base-pcqm4mv1')
    return model