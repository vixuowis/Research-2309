from transformers import pipeline
import torch

# Function to estimate depth in street images
# Uses the Hugging Face Transformers library and a specific model trained for depth estimation
# The model is 'sayakpaul/glpn-nyu-finetuned-diode-221221-102136'
# The function takes in the path to a street image and returns the estimated depth map

def estimate_depth(street_image_path):
    # Create a depth estimation model using the pipeline function
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode-221221-102136')
    # Use the model to estimate depth in the given street image
    depth_map = depth_estimator(street_image_path)
    return depth_map