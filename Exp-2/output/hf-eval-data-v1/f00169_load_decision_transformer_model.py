from transformers import AutoModel


def load_decision_transformer_model(model_name):
    """
    This function loads the Decision Transformer model from the Hugging Face model hub.
    The model is trained on expert trajectories sampled from the Gym Walker2d environment.
    
    Parameters:
    model_name (str): The name of the pretrained model.
    
    Returns:
    model: The loaded model.
    """
    # Load the Decision Transformer model
    model = AutoModel.from_pretrained(model_name)
    return model