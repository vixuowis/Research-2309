from transformers import AutoModel


def load_decision_transformer_model():
    """
    This function loads the pretrained Decision Transformer model for the Gym Hopper environment.
    The model is trained specifically for the Gym Hopper environment, making it suitable for our hopping robot.
    Mean and standard deviation values provided in the API metadata can be used for normalization of the input features, which will help the model generalize better on new robot hopping environments.
    """
    # Load the pretrained model
    decision_transformer_model = AutoModel.from_pretrained('edbeeching/decision-transformer-gym-hopper-medium')
    return decision_transformer_model