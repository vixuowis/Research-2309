from transformers import AutoModel

def load_graphormer_model():
    """
    This function loads the Graphormer model from Hugging Face Transformers.
    The model is pre-trained on the PCQM4M-LSC dataset and is used for predicting molecular properties.
    """
    # Load the pre-trained Graphormer model
    graph_model = AutoModel.from_pretrained('graphormer-base-pcqm4mv1')
    return graph_model