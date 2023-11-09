from transformers import AutoModel

def load_graphormer_model():
    """
    This function loads the pre-trained Graphormer model from Hugging Face Transformers.
    The model is trained on the PCQM4M-LSCv2 dataset for graph classification tasks.
    It can be used for classifying molecular structures in pharmaceutical research.
    """
    # Load the pre-trained model
    model = AutoModel.from_pretrained('clefourrier/graphormer-base-pcqm4mv2')
    return model