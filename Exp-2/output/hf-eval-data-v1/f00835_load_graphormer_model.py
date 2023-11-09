from transformers import AutoModel

def load_graphormer_model():
    """
    This function loads the pretrained Graphormer model 'graphormer-base-pcqm4mv1' from Hugging Face Transformers.
    The model is developed by Microsoft and has won 1st place in the KDD CUP 2021 (quantum prediction track) for predicting molecular properties in the drug discovery domain.
    The loaded model can be used for graph classification tasks, graph representation tasks or fine-tuned on specific downstream tasks in the molecule modeling domain.
    
    Returns:
        model (AutoModel): The loaded Graphormer model.
    """
    model = AutoModel.from_pretrained('graphormer-base-pcqm4mv1')
    return model