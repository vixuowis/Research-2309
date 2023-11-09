from transformers import AutoModel
import torch


def estimate_depth(input_image_path: str) -> torch.Tensor:
    """
    Estimate the depth of a scene in an image using a pretrained model.

    Args:
        input_image_path (str): The path to the input image.

    Returns:
        torch.Tensor: The estimated depth map.
    """
    # Load the pretrained model
    depth_estimator = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221215-095508')

    # Preprocess the input image
    processed_image = preprocess_image(input_image_path)

    # Estimate the depth map
    predicted_depth_map = depth_estimator(processed_image)

    return predicted_depth_map