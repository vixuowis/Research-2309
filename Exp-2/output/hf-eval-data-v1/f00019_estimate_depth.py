from transformers import pipeline
import torch

# Function to estimate depth in images
# This function uses the Hugging Face Transformers library to create a depth estimation model
# The model used is 'sayakpaul/glpn-kitti-finetuned-diode-221214-123047'
# This model is a fine-tuned version of vinvino02/glpn-kitti on the diode-subset dataset
# It is used for depth estimation in computer vision applications
# The function takes an image path as input and returns a depth map

def estimate_depth(image_path):
    # Create a depth estimation model
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-kitti-finetuned-diode-221214-123047')
    # Estimate depth in the input image
    depth_map = depth_estimator(image_path)
    return depth_map