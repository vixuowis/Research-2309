from transformers import AutoModel


def load_pretrained_model():
    """
    This function loads a pretrained model for depth estimation in computer vision.
    The model is loaded from Hugging Face Transformers and is named 'sayakpaul/glpn-nyu-finetuned-diode-221122-082237'.
    This model has been fine-tuned on the diode-subset dataset and can be used for depth estimation tasks.
    """
    # Load the pretrained model
    model = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221122-082237')
    return model