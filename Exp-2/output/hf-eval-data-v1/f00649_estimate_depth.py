from transformers import pipeline
import torch

# Function to estimate depth from a single image
# This function uses the Hugging Face Transformers library to create a depth estimation model
# The model 'sayakpaul/glpn-nyu-finetuned-diode-221122-044810' is loaded, which has been fine-tuned on the diode-subset dataset
# The model is capable of estimating the depth map of a single input image
# The input to this function is an image data
# The output of this function is a depth map

def estimate_depth(image_data):
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode-221122-044810')
    depth_map = depth_estimator(image_data)
    return depth_map