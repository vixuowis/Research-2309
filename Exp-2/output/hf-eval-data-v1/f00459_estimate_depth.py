from transformers import pipeline
from PIL import Image


def estimate_depth(image_path):
    """
    This function estimates the depth map of an image using a pre-trained model from Hugging Face Transformers.
    The model used is 'sayakpaul/glpn-kitti-finetuned-diode-221214-123047', which is trained for depth estimation in computer vision applications.
    The depth map can be used by an autonomous vehicle to plan its navigation path and make proper driving decisions in a parking lot.
    
    Parameters:
    image_path (str): The path to the input image.
    
    Returns:
    depth_map (np.array): The estimated depth map of the input image.
    """
    # Create a pipeline for 'depth-estimation'
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-kitti-finetuned-diode-221214-123047')
    
    # Load the input image
    input_image = Image.open(image_path)
    
    # Estimate the depth map
    depth_map = depth_estimator(input_image)
    
    return depth_map