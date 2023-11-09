from transformers import pipeline
import cv2

# Function to estimate the depth of objects in an image
# captured by a drone's camera
# Uses the 'glpn-nyu-finetuned-diode' model from Hugging Face Transformers
# to estimate depth

def estimate_depth(image_path):
    """
    Function to estimate the depth of objects in an image captured by a drone's camera.
    Arguments:
    image_path : str : Path to the image file
    Returns:
    depth_map : ndarray : Depth map of the image
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Create a depth estimation pipeline
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode')
    
    # Estimate the depth of the image
    depth_map = depth_estimator(image)
    
    return depth_map