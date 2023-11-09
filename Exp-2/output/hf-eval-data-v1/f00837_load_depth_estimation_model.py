from transformers import AutoModel


def load_depth_estimation_model(model_name: str):
    """
    Load a pretrained depth estimation model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pretrained model. For example, 'sayakpaul/glpn-nyu-finetuned-diode-221122-082237'.

    Returns:
        A model object that can be used for depth estimation tasks.
    """
    depth_estimation_model = AutoModel.from_pretrained(model_name)
    return depth_estimation_model